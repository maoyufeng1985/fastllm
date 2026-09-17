# TP4 push all-reduce 落地方案

日期：2026-09-15
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP / CUDA Graph decode
上游证据：`docs/sm70_ar_microbench_deepdive.md` §6、`/home/arproto/ar_phase_probe.cu`、
`/home/arproto/ar_phase_run0.txt`、`/home/arproto/ar_phase_push0.txt`

**状态：暂缓（2026-09-15，决策：人）。** 中心收益 +3%–4% 不值得当前投入。
未写任何引擎代码。重新打开的条件：decode 墙钟重新成为主要目标，或其他
更高收益项做完后仍需要挤 AR。方案本体保留，U0–U4 直接可执行。

> **2026-09-17 更新：代码写完了，自测通过；但真正的加速测试被掉卡打断。**
>
> 用大白话说这件事：这个方案想把四张卡之间"传数据"的方式换一种更省事的做法
> （原来要等对面回应，新的做法是直接把数据塞过去、省掉等待）。我把它写出来了，
> 自测能跑对，但**到底快了多少，还没测出来**。
>
> - **写了什么**：新内核 `FastllmCustomAllReducePushKernel`，位置
>   `src/devices/multicuda/fastllm-custom-allreduce.cu`。四卡版、纯求和（不做
>   residual）、**用 FP32 按卡号顺序相加、最后只转一次精度**——这样结果和现在用的
>   老办法**一模一样**（老的 TP2 版本是用半精度相加，结果会不一样）。
> - **怎么开关**：环境变量 `FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4`，
>   **不设就是老路径**（默认关），所以对现有跑法零影响。
> - **自测**：`customAllReduceRegression` 测试 + `FASTLLM_TEST_CUSTOM_ALLREDUCE_RANKS=4`
>   → **通过**（4 卡、6 条路径全过）。还做了**正控**：开这个开关时日志会打
>   "TP4 push all-reduce engaged"，关掉时一行都不打、测试同样通过——说明新路径
>   确实是**独立**的，没污染老路径。
> - **自己写出来两个真 bug，都修了**：(a) 我在**主机**代码里去读了一个**显存里**的
>   指针，进程直接段错误崩掉（exit 139）——改到有真的主机指针的地方去发；(b) 我传给
>   内核的是**元素个数**，但内核循环数的是 **16 字节一个包**，对 bf16 来说差了 8 倍，
>   读越界（测试报 "expected 40, got 672"）——改成按字节数除以 16。
> - **还有一个更阴的坑：测试显示"通过"，但新代码一次都没跑。** 原因是为 push
>   准备的那块显存缓冲只在"2 张卡"时才分配，四卡时永远是空的，所以我的新代码
>   永远进不去——**而测试照样打 PASS**。我是靠**正控**（去 grep 那行 engaged 日志，
>   发现一行都没有）才抓住的。教训：**"通过"不等于"测到了"**。
> - **已经量到的数**：不开新开关时，8K 输入、C=1，**11.49 毫秒/token（87.04 tok/s）**，
>   首字延迟 3142.75 ms，输出哈希 `d5fcc5fc`——**正好是方案要求的那个哈希**。
>   （方案里写的 76.02 tok/s 是更早的旧基线，不能直接拿来当分母，必须**同场对比**。）
> - **还没量到的数**：打开新开关那次，四张卡**掉线了两次**（都是 `Xid 79/154` + PCIe 报错），
>   **没有数据**。第二次我特意**单独只跑这一条**、而且是在刚重启的干净机器上，结果照样掉卡——
>   所以"就是跑太多四卡测试导致的"这个解释变弱了。现在的相关性是：**开新开关的两次都掉卡，
>   不开的那几次都没事**。这个新做法向其他三张卡直接写数据，正是可能惹毛 PCIe 链路的那类操作，
>   **所以它是主要嫌疑**。但两次样本还不足以断定因果，我在会话内也分不清，**所以不下结论**。
>
> - **处置**：这个开关**保持默认关闭，且未经明确决定不要在这台机器上开启**。

## 1. 目标与指标

主指标：8K C=1 decode 墙钟（TPOP ms/token，3 次取中位数）。
次指标：trace 里 custom AR kernel 时间和；token sha256 必须逐位一致。
判定门：中位数 decode 相对 76.02 tok/s 基线 **≥ +1.5%**，且 sha256 等于
`d5fcc5fc…`，且 `customAllReduceRegression` PASS。三条任一不满足即回滚。

## 2. 基线数字（本轮探针实测）

现役 one-stage（skip-end 后，ping-pong 独立 dest）在 10 KiB、2 block ×
512 thread、128 AR/token：

| 口径 | pull（现役） | push4（探针） |
|---|---:|---:|
| kernel 内合计 | 11.3 µs（barrier 5.1 + payload 6.1） | ~3.6 µs（out 2.05 + poll 1–2） |
| 背靠背每 AR | 14.0 µs | **6.3 µs** |
| 50 µs filler 每 AR | 13.5 µs | **5.3 µs** |
| 数值 | max_abs_err 0 | max_abs_err 0 |

barrier 等待在四卡上均匀（4–5 µs），说明是协议 RTT 而非 skew 吸收，
去掉握手是净赚。这是本方案成立的前提。

## 3. 设计

### 3.1 kernel

新增 `FastllmCustomAllReducePushKernel<T, Ranks>`，把
`FastllmCustomAllReducePushAddKernel`（`fastllm-custom-allreduce.cu:799`，
TP2 专属）的机制推广到 4 rank，去掉 residual-add 语义，变成纯 AR：

1. 每个 thread 认领 ≤1 个 16B 包。先把自己的输入做 `CustomArClearPositiveZero`
   （+0 → −0），`CustomArStoreRelaxedSystem16` 写进 3 个 peer 的
   workspace 槽 `[phase][self]`。
2. 然后对 3 个 peer 的本地槽 `[phase][p]` 用
   `CustomArLoadRelaxedSystem16` 轮询，直到 8 个 half 都非 +0
   （复用 `CustomArIsPositiveZero`，TP2 :850-858 的判据）。
3. 合并必须**按 rank 序**累加：`acc = v(slot0) + v(slot1) + v(slot2) +
   v(slot3)`，p == self 时取本地输入，FP32 累加，最后一次
   downcast。现役 one-stage 就是 ptrs[0..3] 按 FP32 顺序求和后一次
   downcast，两者逐位一致。**探针里"本地先加"的写法不能带进引擎**，
   那只在 self==0 时碰巧等于 rank 序。
4. 写出 dest（本地写，peer 不碰 dest）。
5. **消费后清槽**：`CustomArStoreGlobal16(empty, slot)`（TP2 :870）。
   这是正确性的关键，见 3.3。
6. per-block phase 计数器复用 `CustomArSignal.pushCounter`，kernel 末尾
   `pushCounter[blockIdx.x]++`。双 phase 让连续图回放互不踩踏。

### 3.2 门与 workspace

- 触发条件（全部满足才走 push）：`devices.size() == 4`、`data != dest`
  （ping-pong 已保证）、`!useTwoStage`、`bytes < kCustomArAutoNcclMinBytes`
  （40 KiB，NCCL 硬切以下）、`bytes % 16 == 0`、workspace 分配成功、
  指针注册成功。
- workspace：每卡 `2 phase × 4 slot × kPush4MaxBytes`，
  `kPush4MaxBytes = 40 KiB` → **320 KiB/卡**。init 时一次分配，
  和 `inplaceScratch` 同层（:2343 附近）。
- capture-miss 时返回 false 走现役回退，和 one-stage 一致（:1204-1222）。
- env：`FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4`，缺省 off（0），`1` 强制，
  `auto` 进 auto-test 分档。A/B 期间 off 是默认，落地后再改缺省。

### 3.3 正确性论证（为什么 +0 标记够）

TP2 的协议靠"消费后清槽"闭合：消费者在 call N 把 phase-a 槽清成 +0；
peer 对 phase-a 的下一次写发生在它的 call N+2，而 peer 的 call N+2 在它
call N+1 之后，call N+1 的推送又被消费者 call N+1 的轮询等待兜住。所以
消费者再次轮询 phase-a 时，槽要么是空的（等新数据），要么是新数据，
**两轮之前的旧值不可能存活**。4 rank 下每个消费者清 3 个槽，论证不变。
探针没做清槽，数值校验通过是靠发送方总赢下 ~2 µs 的竞速——**探针结论
只用于收益估计，正确性以引擎实现为准**。

poll 用 relaxed 而非 acquire 是够的：16B 包是单次 store 单 TLP 落地，
看到任一非 +0 即整包可见（TP2 生产验证过的原语，直接复用，不引入我在
探针里用的 `ld.acquire.sys`）。

## 4. 收益（量化）

| 层级 | 现值 | push 后 | 依据 |
|---|---:|---:|---|
| 单 AR kernel | 11.3 µs | ~3.6 µs | 探针 |
| 单 AR 槽（含图开销） | 13.5–14.0 µs | 5.3–6.3 µs | 探针 |
| AR 桶/token（128 次） | ~1.73 ms | 0.68–0.81 ms | 探针 × 128 |
| decode 墙钟 | 13.16 ms（76.02 tok/s） | **12.2–12.96 ms（77.2–82.0 tok/s）** | 见下 |

**2026-09-15 复核（Combine 修复后重测）。** 引擎 AR 实测 slice，按 **device 0**
（四卡里最慢的一卡，决定关键路径）算是 **1.925 ms/token**
（128 × mean 15.04 µs，p50 14.88、p90 19.14、p99 26.71）。此前的
p50 13.95 / mean 14.09 是**四卡混池**的结果：把 device 0/1/2/3 的行按时间
交错后，较快的三卡（mean 14.03–14.22）把中位数拉低。混池对**时长**只是约 5%
的偏差（mean 14.36 vs 15.04，p50 14.08 vs 14.88），但基准应当是 gating 的
device 0。两份 80K trace（修复前/后）的
device-0 时长一致（mean 15.04 / 18.08 µs），所以 15.0 µs 是稳态值；此前引用的
18.07 µs 来自含 warmup 污染的早期工作负载，**是错的，已作废**。

另一个关键量：AR 的 start-to-start 间隔**必须按 device 单独算**。四卡的行共享
时间轴但不共享 stream，交错后得到 p50 **1.8 µs** 的假结论。per-device 的正确
读数是 p50 **87.9 µs**（p90 237.5），即每次 AR 之后还有约 **73 µs** 的其它
kernel（GEMM/attention/norm），AR 并不构成一段纯 AR 串行区。分布是双峰的：
20.5% 的间隔在 60–80 µs、65.1% 在 80–100 µs、13.3% 在 200–500 µs。所以 push
省下的时间能否暴露，取决于它落在哪一段，不能指望"把 AR 串起来跑"。

AR 的绝对字节数不随上下文变，所以它的占比在三个口径下是（每行用各自 trace 的
device-0 mean）：

| 口径 | device-0 AR mean | AR 桶 | token | AR 占比 |
|---|---:|---:|---:|---:|
| 8K C=1 | 14.65 µs | 1.876 ms | 11.45 ms（87.4 tok/s） | **16.4%** |
| 80K C=1 | 15.04 µs | 1.925 ms | 13.65 ms（73.2 tok/s） | **14.1%** |

按探针 5.3 µs/AR 落地的上限推（各用自身 mean）：8K 省
(14.65 − 5.3) × 128 = **1.197 ms/token** → 87.4 → 97.6 tok/s（+11.7%）；
80K 省 (15.04 − 5.3) × 128 = **1.247 ms/token** → 73.2 → 80.6 tok/s（+10.1%）。
这是**上限**，不是预测：skip-end 那轮 kernel 省 640 µs 只兑现 150 µs
（23%），因为 barrier 等待被排队吸收；push 省的是真实工作（flag RTT 加远端
读轮询改本地轮询），兑现率应显著高于 23%。**中心估计 +4%–6%。**
上限的物理封顶：AR 桶不可能低于 128 × 5.3 µs ≈ 0.68 ms。

范围之外不受益：≥40 KiB 已硬切 NCCL（two-stage 门在 512 KiB）；
PairAdd 结构性不可行（attn AR 经 RMSNorm/gateup/swiglu/down 才到 MLP AR，
`qwen3_5.cpp:11780-11820`）；MTP 路径 in-place 不 qualifying，保持原样；
其他模型（hy_v3、step3p5 等）调用仍 in-place，push 对它们休眠，零风险。

代价：+320 KiB/卡显存；kernel 代码 ~200 行。

## 4.1 引擎实测口径（2026-09-15，80K C=1 trace 逐 kernel 统计）

`/home/nsys/dec80k_post.sqlite`，device 0，decode 窗口 1602.8 ms：

- AR kernel 时长 **p50 14.88 / p90 19.14 / p99 26.71 µs**，mean 15.04；128 次/token。
- AR 桶 **1.925 ms/token**（128 × 15.04 µs），占窗口墙钟 **13.5%**、占 busy **14.3%**。
- AR 的 start-to-start 间隔（**per-device**）**p50 87.9 µs**（p90 237.5），即每次
  AR 后面平均有约 73 µs 的其它 kernel（GEMM/attention/norm）。所以 AR 不构成
  一段纯 AR 串行区，push 省下的时间是否暴露，取决于它是否落在关键路径上，而不是靠
  "把 AR 串起来跑"。**注意**：不按 device 拆开、把四卡的行交错后算，会得到
  p50 1.8 µs 的假"背靠背"读数（四卡共享时间轴但不共享 stream），量 headroom
  前必须先拆卡。
- 路由正确：整个 decode 窗口 **0 次 NCCL all-reduce**，14336 次全是 custom。
  10 KiB 的 auto 分档在本 workload 上生效。

## 5. 实施单元（每个独立可验证）

1. **U0 正确性预检（已完成）**：TP2 kernel 的清槽与判据读完（:850-870），
   本文件 §3.3 即结论。无代码。
2. **U1 kernel + 门**：按 §3.1/§3.2 实现，env 缺省 off。
   验证：编译 + `customAllReduceRegression` 扩一个 TP4 push 用例
   （eager 单次 + 512 次错位图回放，对照 NCCL 数值，复用现有
   `RunGraphPingPongOrderingRegression` 的骨架，dest 独立）。
3. **U2 数值逐位验证**：引擎 decode 打开 push4，8K C=1 128 token，
   token sha256 必须 == `d5fcc5fc…`。不等则修求和序或变换，重跑 U2，
   不进 U3。这是位级 parity 的硬门。
4. **U3 A/B 计时**：8K C=1 128 token，off/on 各 3 次取中位数，
   对照 76.02 tok/s。过 §1 判定门则保留并提交（只 stage 改动的文件）；
   不过则 env 回 off、代码回滚，记录一行。
5. **U4 auto 分档**（U3 过门后）：auto-test 在 10 KiB 加测 push vs
   one-stage，胜 ≥3% 才把 small 路径切到 push；`auto` 成为缺省。
   验证：auto-test 日志 + 回归。

## 6. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 旧槽伪到货（探针未清槽的坑） | 消费后清槽（TP2 生产验证，:870）；U2 位级 hash 门兜底 |
| 某卡挂死导致全卡轮询死等 | 与现役 barrier spin 同一失败面，不引入新风险 |
| 求和序导致 1 ulp 差、hash 变 | §3.1 第 3 条 rank 序 FP32；U2 硬门 |
| 图回放踩槽 | 双 phase + per-block 计数（TP2 同款） |
| 显存 | +320 KiB/卡，可忽略 |
| push 不赢 | env 缺省 off，回滚 = 还原 kernel 文件与门 |

## 7. 复现

```sh
cd /home/arproto
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_70 \
    --default-stream=per-thread --expt-relaxed-constexpr \
    -Wno-deprecated-gpu-targets -o ar_phase_probe ar_phase_probe.cu -lcudart -lpthread
./ar_phase_probe 0 0      # pull 背靠背：14.0 µs/AR
./ar_phase_probe 0 1      # push 背靠背：6.3 µs/AR
./ar_phase_probe 50000 0  # pull 引擎形态：13.5 µs/AR
./ar_phase_probe 50000 1  # push 引擎形态：5.3 µs/AR
```

引擎 A/B：

```sh
FASTLLM_PAGED_CUBLAS_CHUNK=2048 PYTHONPATH=/home/fastllm/build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 1 --tokens 16384 --dtype auto \
  --enable_thinking false --prefix_cache false --input_tokens 8192 \
  --output_tokens 128 --batch 1 --warmup 1 --temperature 0 --top_k 1
# 对照 FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4=0/1
```
