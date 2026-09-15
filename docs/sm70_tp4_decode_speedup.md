# TP4 decode 加速方案

日期：2026-09-14
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP / C=1
口径：稳态 decode 墙钟。prefill 调度见 `docs/sm70_long_prefill_chunk_plan.md`，
本文件不碰。

---

## 0. 一句话

不要移植 1Cat pack32。TP4 decode 已经在跑 FastLLM custom one-stage + CUDA Graph。
还活着的杠杆只有三件：**图内独立 dest 关掉 end barrier**、**GDN-in 走 QPN2**、
**C>1 时 40 KiB+ 走 NCCL**。第一件最大、也最难接线。

## 1. 现在每 token 花在哪

干净稳态 decode（`/tmp/nsys_dec.sqlite` 最后一段，12.75 ms，与 13.31 ms 基线差 4%）：

| 组件 | µs/token | 占比 | 还能不能动 |
|---|---:|---:|---|
| QPN2（208 条过门投影） | 5290 | 41.5% | N=5120 已 85% HBM，余量在 N=8704 / N=3584 |
| custom AR × 128 | 2725 | 21.4% | **主杠杆**。21.3 µs/次含排队 |
| attention（QGateKV + split + combine） | 1534 | 12.0% | 8K 固定开销；80K 另测 |
| GDN-in TurboMind + crop × 48 | 1422 | 11.1% | 形状已 pad 到 4128，缺接线 |
| lm_head GEMV | 718 | 5.6% | 98% HBM，不动 |
| RMSNorm | 525 | 4.1% | 不动 |
| 其余 | ~540 | 4.3% | 含 SwiGLU 123 µs |

8K C=1 已经 **75 tok/s**，1Cat ~71。目标不是追上 1Cat，是把本机还空着的 1–2 ms
拿回来。上限大约：

| 项 | 探针/账本上限 | 墙钟预期 |
|---|---|---|
| no-end（独立 dest） | 单次 21.3 → ~17 µs 量级；探针 12.94 vs graph NCCL 16.05 | **未知**。排队可能把 3 µs 吃掉。先接线再量 |
| GDN-in → QPN2 | 1422 → ~730 µs | **~0.6–0.7 ms，+5%**。设计已有 |
| QPN2 战术（QPN4 几何） | N=8704/3584 拉到 769 GB/s | ~0.7 ms，**赌注**，先 A/B |
| fused SiLU | 123 µs | ≤1%，不做 |
| 40 KiB+ NCCL | C≥4 才碰到 | C=1 decode 10 KiB 用不上 |
| pack32 | 10 KiB 24.4 vs 现役 18.0 | **负收益，禁止** |

## 2. 现役路径（不要再猜）

```
TP4 Qwen3.5 decode 10 KiB, graph on
  addPartialToResidualReduce
    TryTP2P2PAllReduceAdd  → TP=2 only, miss
    rank0 AddTo / 其他 CopyFrom
    FastllmNcclAllReduce(hidden, hidden)     // dest == src
      auto: custom one-stage
        start barrier (st.release.sys + ld.acquire.sys)
        16 B pack, grid = ceil(packed/512) = 2 CTA
        writeAfterBarrier  (结果在寄存器，end 后再写回)
        end barrier
```

打到这条路径的优化已经在树上：FastLLM barrier、2 CTA 几何、graph 指针预注册、
fused copy-back、10 KiB auto 探针、≥40 KiB 硬切 NCCL。

走不到的：PushAdd（`devices.size()==2`）、PairAdd（MoE 双张量；attn-out 与 MLP
有数据依赖）、two-stage（≥512 KiB）、`NcclSubmitRendezvous`（奇数 TP）。

128 次/token = 64 层 × (attn-out AR + MLP AR)。相邻两次不能 PairAdd。

## 3. 方案，按做的顺序

### P0 — 图内 no-end（改图，不改 packing）

**为什么是 P0。** 探针已经过门：CUDA Graph、per-thread stream、独立 dest，
10 KiB no-end 12.94 µs vs graph NCCL in-place 16.05 µs（−19%）。512 次错位回放
4 卡一致。障碍不是 kernel，是调用：

```
qwen3_5.cpp:11324 / 12657
FastllmNcclAllReduce(hidden.cudaData, hidden.cudaData, ...)
```

`data == dest` 时 `writeAfterBarrier` 必须开，否则 peer 还在读就被覆盖。
关掉 end barrier 的合法条件是 **dest 不是任何 rank 的 input**。

**落地步骤。**

1. 给每个 AR 站点一块常驻 dest，大小 = hidden（5120 × 2 B = 10 KiB），
   按 device 预分配，capture 前建好。不要在 capture 里 `cudaMalloc`。
2. 调用改成 `FastllmNcclAllReduce(src, dest, ...)`，随后 `hidden` 的后续算子
   读 dest。最省事：AR 后 `hidden.cudaData` 与 dest 交换指针（图内指针必须
   稳定——所以不能每步 malloc，只能预分配两块轮换，或让下一算子直接吃 dest）。
3. `RunCustomArCandidate`：`data != dest` 时关 `writeAfterBarrier`。
   现有 fused-copyback 路径保持给仍 in-place 的调用。
4. graph capture 必须看到 dest 指针。warmup 注册 custom AR 的 pointer tuple
   时把 dest 也注册进去。
5. 正确性：greedy 8K C=1，`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`（纯 NCCL）与
   no-end custom 的 token sha256 相同。4 卡逐元素对 NCCL。
6. 墙钟：同一 8K C=1 graph-on 基线（现在 13.31 ms / 75 tok/s）。
   **探针 19% 不是墙钟。** 引擎观测 20–69 µs 含排队；若排队主导，墙钟可能
   <5%。过门标准：decode ≥ +3% 且 sha256 不变，否则回滚留 env。

**不要做。** 不要为了 no-end 去改 barrier 指令；不要把 PushAdd 的门扩到 TP4
（Lamport push 的 workspace 与 4 卡握手是另一套，没有本机数字）。

**风险。** 双缓冲 10 KiB × 层数不是问题（128 × 10 KiB = 1.25 MiB/rank）。
真正的风险是图内后续 RMSNorm / Linear 仍绑着旧 `hidden` 指针。必须沿
`addPartialToResidualReduce` 的两个调用点（attn-out、MLP-out）把消费者改完，
MTP 旁路（`29591` / `29818`）no-MTP 可以先不改。

### P1 — GDN-in 走 QPN2

48 层 `in_proj_qkvzba`，本地 N=4120，现役 TurboMind + `CropNvfp4Output`，
1422 µs/token。侧车按 4128 建、GEMM 写 4120、复用现有 crop。设计在
`docs/sm70_qpn_npad_design.md`，改动比当初估计小：kernel 不用改，改
`Nvfp4QpnCanRun` 的 N 门和 warmup 建侧车。

过门：`FASTLLM_SM70_NVFP4_QPN2=0` 的 A/B 方向反过来——打开 GDN-in QPN2 应
接近 −0.6 ms，sha256 不变。显存：侧车 +0.57 GB/rank，16 GB 上要重启后
`nvidia-smi` 实测，不能靠估算。

### P2 — 只在有数字以后才动 QPN2 几何

N=8704（MLP gate/up）600 GB/s、N=3584（QKV）382 GB/s，相对 N=5120 的 769 GB/s
有缺口。QPN4 与 QPN2 **同门**（实际 K%128、N%32、M=1），不扩覆盖面。
收益全是发射几何赌注。先做算子级 A/B：同一权重、同一 M=1，QPN4 vs 现役 QPN2。
赢 ≥3% 再接线 fused SiLU；输了停。fused epilogue 单独只值 0.12 ms。

### 明确不做

| 项 | 原因 |
|---|---|
| 1Cat pack32 / `block_limit=1` | 10 KiB 24.4 vs 18.0，barrier 本身慢 36% |
| 关 `NcclSubmitRendezvous` | TP4 不构造 |
| decode 10 KiB 换 NCCL | graph 下与 custom 打平（20.1 vs 19.9 µs） |
| PairAdd 合 attn+MLP | 数据依赖，dest 不独立时非法 |
| PushAdd 扩到 TP4 | 无本机 4 卡数字；workspace 契约不同 |
| QPN8 | 要 FP8 权重契约，当前 checkpoint 是 NVFP4 |
| XQA / D256 workspace | 8K attention 12%，不是第一刀；80K 另测 |
| mixed batch / 入队分块 | 并发问题，见 long-prefill plan |

40–512 KiB 硬切 NCCL **已经落地**（`kCustomArAutoNcclMinBytes = 40 KiB`）。
C=1 decode 用不上。C=4/8 的 40/80 KiB 消息会自动走，不必再做一项。

## 4. 执行顺序

```
PR-D1  独立 dest + no-end 接线（qwen3_5 addPartialToResidualReduce × 2）
       正确性门：greedy sha256
       速度门：8K C=1 graph-on ≥ +3%
       失败：env 回滚，留下 dest 分配（后续算子仍可读 in-place）

PR-D2  GDN-in QPN2 侧车（按 qpn_npad_design）
       与 D1 正交，可并行
       速度门：~0.6 ms；显存门：16 GB 不 OOM

PR-D3  QPN2 vs QPN4 算子 A/B
       只在 D1/D2 落地后、且 8K 仍觉得 compute 紧时做
```

D1 的测量必须用引擎 decode，不要再用 `ar_bench` 外推墙钟。
建议同一命令：入 2048 / 出 32 / batch 1 / graph on，报 TPOP 与 sha256。

## 5. 预期墙钟（诚实区间）

基线 13.31 ms / 75.1 tok/s。

| 组合 | 乐观 | 保守 | 依据 |
|---|---|---|---|
| 只 D2 | 12.6 ms / 79 | 12.7 ms / 79 | 账本 0.6–0.7 ms，A/B 口径可靠 |
| D1+D2 | 12.0–12.6 ms / 79–83 | 12.5–12.7 ms / 79 | D1 可能被排队吃掉 |
| 再加 D3 | 再 0–0.7 ms | 0 | 无本机证据 |

达不到「AR 减半 → +15%–19%」。那是把 1Cat 的 12.4 µs 预算套到本机 21.3 µs
观测值上，且没把排队算进去。本方案能捍卫的是 **D2 的 +5%**；D1 是唯一还可能
把 AR 再往下压的改图实验，必须用墙钟说话。
