# TP4 push all-reduce 落地方案

日期：2026-09-15
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP / CUDA Graph decode
上游证据：`docs/sm70_ar_microbench_deepdive.md` §6、`/home/arproto/ar_phase_probe.cu`、
`/home/arproto/ar_phase_run0.txt`、`/home/arproto/ar_phase_push0.txt`

**状态：暂缓（2026-09-15，决策：人）。** 中心收益 +3%–4% 不值得当前投入。
未写任何引擎代码。重新打开的条件：decode 墙钟重新成为主要目标，或其他
更高收益项做完后仍需要挤 AR。方案本体保留，U0–U4 直接可执行。

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

墙钟换算说明：skip-end 那轮 kernel 省了 ~640 µs，墙钟只兑现 150 µs
（兑现率 23%），因为 barrier 等待被排队吸收。push 不同：省的是真实工作
（flag RTT + 远端读轮询改本地轮询），不是等待，兑现率应显著高于 23%。
**中心估计 +3%–4%（~0.4–0.5 ms/token，77.5–78.5 tok/s）**，区间
+1.5%–+7.7%。上限的物理封顶：AR 桶不可能低于 128 × 5.3 µs ≈ 0.68 ms。

范围之外不受益：≥40 KiB 已硬切 NCCL（two-stage 门在 512 KiB）；
PairAdd 结构性不可行（attn AR 经 RMSNorm/gateup/swiglu/down 才到 MLP AR，
`qwen3_5.cpp:11780-11820`）；MTP 路径 in-place 不 qualifying，保持原样；
其他模型（hy_v3、step3p5 等）调用仍 in-place，push 对它们休眠，零风险。

代价：+320 KiB/卡显存；kernel 代码 ~200 行。

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
