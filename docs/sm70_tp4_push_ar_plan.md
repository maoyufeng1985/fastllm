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
> - **已经量到的数（修正后）**——同一台机器、同一条命令、开/关各跑一次：
>
>   | | 关掉开关 | 打开开关 |
>   |---|---:|---:|
>   | 每个 token 耗时 | 11.48 毫秒 | **10.81 毫秒** |
>   | 每秒生成 | 87.11 个 token | **92.51 个 token** |
>   | 总时间 | 4.6020 秒 | **4.5258 秒** |
>   | 输出哈希 | `d5fcc5fc…` | `d5fcc5fc…`（**一字不差**）|
>   | 新代码是否真的跑了 | 否（日志 0 行）| **是（日志 1 行）** |
>
>   → **快了大约 6%，结果一字不差，显卡没有掉线、没有新的报错。**
>   方案要求的三条（速度达标、哈希一致、自测通过）**都满足**。
>
>   **但要注意样本量**：每边只跑了 1 次，方案原本要求"跑三次取中间值"。
>   这台机器有"一次开机最多跑两个四卡测试"的限制，要凑够 3 次得**分几次开机**。
>   所以这个 +6% 目前是**一次测量的结果**，还不是稳定的中位数。
>
> - **撤错：之前说"打开开关那次掉卡，所以它是嫌疑"，这个判断作废。**
>   那两次跑的是**没有重新编译的旧版本**——我改了代码，却只重编了自测程序，
>   没重编跑分程序真正加载的那块库（两个是不同的文件）。旧版本里我新写的代码
>   因为一个门没通过，**根本进不去**，日志里"engaged"一行都没有。重新编译之后，
>   同一条命令跑通了、显卡也没事。所以**那两次掉卡不能算在它头上**。
>   （掉卡的真实原因仍未确定，这台机器本身有掉卡历史，本轮不再追。）
>
> - **处置**：开关**保持默认关闭**（不设它就走老路径）；要开启需明确决定。
>   目前的实测结论是"**有收益、可用**"，但样本只有一次，建议多跑几次开机再改默认值。

> **2026-09-18 更新：3 次关 + 3 次开跑完了，+5.84%；同时补上了新路径自己的计数。**
>
> - **补上"新路径服务了多少次"这个计数**（之前缺）。新增普查槽
>   `kCensusPush4Launched`（`fastllm-custom-allreduce.cu:1750-1761`），push 内核成功
>   发射时记一次（`:1892-1895`），退出时打印一行
>   `launched on the TP4 push kernel`。**为什么必须单独一格**：原来的
>   `kCensusLaunched` 把 push 与一步式内核记在同一格，分不开；按项目规约
>   （改了代码看收益，必须给出只属于新路径的计数且 > 0），没有它"开关开了"和
>   "新代码真的跑了"在日志上长得一模一样。
> - **复核 2026-09-17 那次对照，发现它两个变量一起动了。** 关掉开关那次没设
>   `FASTLLM_CUDA_CUSTOM_ALLREDUCE`（走 auto，日志显示 6 条精度/尺寸路径只启用 3 条），
>   打开开关那次设了 `=1`（forced，6 条全开，并且关掉了"≥40 KiB 交给 NCCL"那条规则）。
>   逐条核对日志后：**10 KiB 那档两次都在走 custom 内核**（auto 自测把 fp16/bf16/fp32
>   small 三条都标成 enabled），**预填充两次都走 NCCL**（消息几十 MiB，超过自定义路径
>   8 MiB 上限；首字延迟 3143.85 对 3152.91 ms 印证）。所以那次在 decode 上仍然是
>   push 对 pull 的对比，结论没被推翻；但为了不再留这个口子，本次重测**两组都设 forced**。
> - **判定门里那个"76.02 tok/s 基线"是旧数，作废。** 当前同配置、关掉开关的中位是
>   **87.18 tok/s**（本次实测）。及格线改成"与**同场**关掉开关的中位比 ≥ +1.5%"，
>   不再引用跨日期的常量（旧数字来自 ping-pong skip-end 的中间态，见
>   `sm70_ar_microbench.md:146-147`）。
> - **重测：3 次关、3 次开，交错顺序（off on on off off on），每轮自验。**
>   脚本 `tools/push4_ab.sh`，`PROFILE=bench`（历史那套旗标）。
>
>   | 轮次 | 开关 | Total (s) | TPOP (ms/token) | TTFT (ms) | sha256 | push4 计数 | 自验 |
>   |---|---|---:|---:|---:|---|---:|---|
>   | 1 | 关 | 4.6094 | 11.49 | 3149.78 | `d5fcc5fc` | 0 | OK |
>   | 2 | 开 | 4.5189 | 10.81 | 3145.48 | `d5fcc5fc` | 4096 | OK |
>   | 3 | 开 | 4.5152 | 10.80 | 3143.57 | `d5fcc5fc` | 4096 | OK |
>   | 4 | 关 | 4.6032 | 11.47 | 3146.23 | `d5fcc5fc` | 0 | OK |
>   | 5 | 关 | 4.6004 | 11.47 | 3144.01 | `d5fcc5fc` | 0 | OK |
>   | 6 | 开 | 4.5110 | 10.80 | 3139.99 | `d5fcc5fc` | 4096 | OK |
>
>   中位：关 **11.47 ms/token（87.18 tok/s）**，开 **10.80 ms/token（92.59 tok/s）**，
>   **每个 token 少 0.67 ms，快 5.84%（吞吐 +6.2%）**。两组区间不重叠
>   （关最慢 11.47 > 开最快 10.81）。首字延迟两组几乎一样
>   （3139.99–3149.78 ms），与"push4 只作用于小于 40 KiB 的吐字消息"一致。
> - **三条自验全过，反向检查也在**：打开开关的三次 `push4=4096`，关掉的三次
>   `push4=0`；六次的"launched on the custom kernel"都是 **6144**，说明两组做的
>   自定义求和次数完全相同，只有内核不同。跑完 dmesg 的 Xid 计数 413 → 413（无新增），
>   四卡显存归零。
> - **读这些计数的边界**：普查统计的是**整个进程**（含模型加载与 warmup），
>   不能当"每 token 多少次"读。4096/6144 这个比例只说明 push4 覆盖了本次三分之二的
>   自定义求和，另外那 2048 次每次约 90 KB，超过 push4 的 40 KiB 门，仍走一步式内核。
> - **生产旗标档的对照已补**（`PROFILE=prod`，8K，3 关 3 开）：中位 12.16 → 11.67
>   ms/token，**−4.03%**，方向不变、幅度比 bench 档小。上面那个 +5.84% 是在历史的
>   bench 旗标下测的，**不能直接当生产的收益**；两档并列见 §8，长上下文见 §8.1/§8.2。

## 1. 目标与指标

主指标：8K C=1 decode 墙钟（TPOP ms/token，3 次取中位数）。
次指标：trace 里 custom AR kernel 时间和；token sha256 必须逐位一致。
判定门：中位数 decode **相对同场、同旗标下关掉开关的中位 ≥ +1.5%**，且 sha256 等于
`d5fcc5fc…`，且 `customAllReduceRegression` PASS。三条任一不满足即改回默认关闭。
（原判定门写的是"相对 76.02 tok/s 基线 ≥ +1.5%"；那个常量来自 ping-pong skip-end
的中间态，已作废，见 2026-09-18 更新。）

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

现成脚本（比手敲可靠，逐轮自验、掉卡即停、DRYRUN 不碰显卡）：

```sh
tools/push4_ab.sh                      # bench 档，6 次（3 关 3 开，交错）
PROFILE=prod       tools/push4_ab.sh   # 生产服务那套旗标
PROFILE=prod16k    tools/push4_ab.sh   # 生产旗标 + 输入 16K
DRYRUN=1 tools/push4_ab.sh             # 只看命令行
```

---

## 8. 生产旗标档的对照（2026-09-18）

`§2026-09-18 更新` 里那个 +5.84% 是在历史的 bench 旗标下测的（`--cuda_embedding
--max_batch 1 --tokens 16384`），**不是生产的旗标**。生产服务跑的是另一套
（`--low_gpu_mem --kv_cache_dtype fp8_e4m3 --prefix_cache true --chunked_prefill_size
8192 --max_batch 8`，KV 池自动定容）。所以另跑了一组同形状的对照。

**转移检查（把 bench 档的数搬到生产档之前必须写）**：

1. `原测量对象` = Qwen3.8-27B-QUASAR-NVFP4 / 4×V100-SXM2 / TP4 / bench 旗标 / 8K 入 128 出 C=1 / 3 关 3 开。
2. `现测量对象` = 同一模型同一机器同一卡数，**换成生产服务那套旗标** / 8K 入 128 出 C=1 / 3 关 3 开。
3. `两者关系` = **不同对象**（旗标不同，是两次独立测量）。可比的理由：同一天、同一份库、
   同一台机器、同样的输入输出长度、同样的交错顺序（off on on off off on）与同样的四条自验；
   唯一变量是旗标。所以两者只能并列报，不能用其中一个去推另一个。

**产物**：`/tmp/push4_bench_*.out`（bench 档 6 次）、`/tmp/push4_prod_*.out`（prod 8K）、
`/tmp/push4_prod16k_*.out`（prod 16K）、`/tmp/push4_prod32k_*.out`（32K，六次全废）；
每次的跑分输出同名，看门狗诊断在同名 `.gpuwatch.txt`。
**库的版本**：bench 与 prod 两档用的是上午那份库；16K 档与自测用的是同一天重编后的库，
两次之间只差普查报告表头的一处文字（`per rank` → `whole process, all ranks`），
计数与耗时不受影响。

| profile | 关掉开关中位 | 打开开关中位 | 每 token 省 | 相对变化 | 关掉那组 tok/s → 打开那组 tok/s |
|---|---:|---:|---:|---:|---:|
| `bench`（8K，历史旗标） | 11.47 ms | 10.80 ms | 0.67 ms | **−5.84%** | 87.18 → 92.59 |
| `prod`（8K，生产旗标） | 12.16 ms | 11.67 ms | 0.49 ms | **−4.03%** | 82.24 → 85.69 |

两次的样本区间都不重叠（bench：关最慢 11.47 > 开最快 10.81；prod：关最慢 12.09 >
开最快 11.91）。prod 档的逐轮数据：

| 轮次 | 开关 | Total (s) | TPOP (ms/token) | TTFT (ms) | sha256 | push4 计数 | 自验 |
|---|---|---:|---:|---:|---|---:|---|
| 1 | 关 | 2.4483 | 12.09 | 912.29 | `d5fcc5fc` | 0 | OK |
| 2 | 开 | 2.4249 | 11.91 | 911.72 | `d5fcc5fc` | 98304 | OK |
| 3 | 开 | 2.3938 | 11.67 | 911.96 | `d5fcc5fc` | 98304 | OK |
| 4 | 关 | 2.4528 | 12.16 | 908.43 | `d5fcc5fc` | 0 | OK |
| 5 | 关 | 2.4651 | 12.24 | 910.32 | `d5fcc5fc` | 0 | OK |
| 6 | 开 | 2.3860 | 11.63 | 909.15 | `d5fcc5fc` | 98304 | OK |

读法：**同一份代码换到生产旗标下，收益从 5.84% 降到 4.03%，方向不变。**
首字延迟两组一致（908.43–912.29 ms），说明打开开关没有动预填充。
六次 sha256 与 bench 档也一致（`d5fcc5fc`），即两套旗标产出同一条贪婪流。

**生产档 decode 比 bench 档慢 0.69 ms/token（12.16 对 11.47，慢 6.0%）**，这是同一份库
在两种旗标下的实测差。哪几个旗标造成的**没有拆开测**，候选是 `--low_gpu_mem`
（关掉 GPU token 交接与 CUDA embedding）与 `--kv_cache_dtype fp8_e4m3`，未验证。

### 8.1 长上下文那一档：第一次跑废了，第二次才成

生产那条会话的上下文约 28K，8K 不能直接代表它，所以加了一档 32K 输入。
**那一档六次全部吐 0 个 token**（`Actual output tokens 0`，Total 约 1.03 s，
TTFT/TPOP 都是 `n/a`），令牌流哈希六次一致（`cbe5cfdf`）。

**这是个假通过，值得记下来**：0 token 的那几轮里 `Total time` 在、哈希六次一致，
按原来的三条判据会全部判成 OK，然后我就会拿一个"总时间 1.03 s"去算收益。
现在脚本加了第四条：**TPOP 必须是数字，否则作废**（`tools/push4_ab.sh` 的 C4）。

**降一档到 16K 输入后正常出字**（单跑探针：128 token、TTFT 923.30 ms、
TPOP 12.64 ms/token、哈希 `48bb3034`；换输入长度换哈希是应该的）。
16K 档的 3 关 3 开对照结果见 §8.2。

**这一档没有证明的事**：32K 输入为什么吐 0 个 token **没查**（生产服务在 28K 上下文上
正常出字，所以不像引擎的通病，更像基准程序在这个长度上的行为），也没有测 28K/80K 的对照。

### 8.2 16K 输入（生产旗标）的对照

| profile | 关掉开关中位 | 打开开关中位 | 每 token 省 | 相对变化 | 关掉那组 tok/s → 打开那组 tok/s |
|---|---:|---:|---:|---:|---:|
| `prod`（8K 输入） | 12.16 ms | 11.67 ms | 0.49 ms | **−4.03%** | 82.24 → 85.69 |
| `prod16k`（16K 输入） | 12.51 ms | 12.04 ms | 0.47 ms | **−3.76%** | 79.94 → 83.06 |

`prod16k` 逐轮：

| 轮次 | 开关 | Total (s) | TPOP (ms/token) | TTFT (ms) | sha256 | push4 计数 | 自验 |
|---|---|---:|---:|---:|---|---:|---|
| 1 | 关 | 2.5272 | 12.62 | 923.99 | `48bb3034` | 0 | OK |
| 2 | 开 | 2.4484 | 11.99 | 925.08 | `48bb3034` | 98304 | OK |
| 3 | 开 | 2.4551 | 12.04 | 925.68 | `48bb3034` | 98304 | OK |
| 4 | 关 | 2.5126 | 12.51 | 923.28 | `48bb3034` | 0 | OK |
| 5 | 关 | 2.5132 | 12.51 | 924.55 | `48bb3034` | 0 | OK |
| 6 | 开 | 2.4603 | 12.11 | 921.74 | `48bb3034` | 98304 | OK |

区间不重叠（关最慢 12.51 > 开最快 12.11）。输入从 8K 拉到 16K，关掉开关那组每 token
从 12.16 涨到 12.51 ms（+0.35 ms，注意力随上下文变长），**打开开关那组省下的绝对时间
几乎不变**（0.49 → 0.47 ms），比例因此从 4.03% 降到 3.76%。

**汇总三档**：bench 8K −5.84%、prod 8K −4.03%、prod 16K −3.76%。方向三次一致；
生产旗标下比历史 bench 旗标下小一档。

### 8.3 自测及格线（`customAllReduceRegression`）的实测状态

方案的及格线里有一条 `customAllReduceRegression` PASS。本轮实测：

| 配置 | 结果 | 引擎自测报的 10 KiB 热路径（eager） |
|---|---|---|
| **PUSH4 开**（本次改动） | **PASS**（ranks=4、enabled=1、selected_paths=3、tested_paths=3，3 行 engaged） | custom **13.763** µs 对 NCCL 14.776 µs → **custom enabled** |
| PUSH4 关（本次改动） | **FAIL**：`policy mismatch: expected enabled=1, got 0`；`auto mode retained NCCL for every tested path` | custom 17.203 µs 对 NCCL 16.589 µs → **NCCL retained** |
| PUSH4 关，**改动前**的库（把本次改动 stash 掉重编，跑两次） | **两次都 FAIL，同一句话** | custom 17.285 µs 对 NCCL 16.814 µs → NCCL retained |

**三条读法：**

1. **这个 FAIL 不是本次改动造成的**（已验证：同一台机器、同一条命令、改动前的库
   连跑两次同样 FAIL，报同一句话；两条 custom 时间 17.285 对 17.203 µs，差 0.5%，
   在噪声内）。
2. **PUSH4 让这条自测从 FAIL 变 PASS**，靠的是它自己的数字：同样 eager、同样 10 KiB，
   custom 从 17.203 掉到 **13.763 µs（快 20%）**，于是从"比 NCCL 慢 3.7%"变成
   "比 NCCL 快 6.9%"，策略才把 custom 打开。
3. **现状要说清**：本机当前**不带这个开关**时，引擎自评会把自定义求和全部让给 NCCL
   （eager 档），于是自测报 `enabled=0` 与用例期望的 `enabled=1` 不符而失败。
   这是本次改动之前就存在的状态，与 push4 无关，但它意味着
   **方案第 1 节那条及格线在缺省配置下今天过不了**，要单独处理（改用例期望，或查
   eager 档为什么比 NCCL 慢）。
