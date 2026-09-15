# 1Cat 优化算法移植：按实测收益排序

日期：2026-09-14
范围：Qwen3.8-27B-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP
方法：先测 FastLLM 自己的时间分布，再决定移植什么。不照搬 1Cat 的排序。

## 0. 为什么不能直接照搬 1Cat 的排序

1Cat 的性能分解是在 **vLLM + Flash-V100/E4M3 XQA** 里测的。FastLLM 用的是
native paged attention，引擎、调度、attention 后端都不同。同一份代码在两个
引擎里的占比可以差一个数量级。所以本文的排序依据是**本机实测**，不是 1Cat
的表格。

## 1. 实测：FastLLM 的 decode 时间分布

### 1.1 kernel 级（nsys，两次独立抓取，占比高度一致）

| kernel | 占比 A | 占比 B |
| --- | ---: | ---: |
| `ncclDevKernel_AllReduce_Sum_f16_RING_LL` | 39.6% | 38.1% |
| `cutlass_70_tensorop_h884gemm_128x128_tn_align8` | 25.0% | 23.3% |
| TurboMind `gemm_kernel`（三个变体合计） | 16.1% | 17.1% |
| `FastllmCudaNVFP4Block162HalfKernel`（NVFP4→FP16 反量化） | 3.8% | 4.4% |
| `volta_h884gemm_64x64` 等 FP16 GEMM | 4.4% | 4.5% |
| **QPN2（`nvfp4_qpn2_sm70_kernel`，原生 NVFP4）** | **1.4%** | **1.5%** |
| `FastllmCustomAllReduceKernel` | 1.2% | 1.4% |
| attention 相关（TransferAttn + paged softmax + combine） | ~1.5% | ~1.5% |

两次的 workload 差别很大（一次 2048 入 / 32 出，一次 256 入 / 256 出），
占比却几乎一样，说明这个分布是稳定的，不是采样噪声。

**最关键的一行是 QPN2 只占 1.4%–1.5%。** 也就是说原生 NVFP4 路径目前只覆盖
很小一部分投影，大部分量化权重的计算走的是「反量化成 FP16 再跑 FP16 GEMM」
或 TurboMind，而不是 1Cat 那样全程原生 NVFP4。

### 1.2 组件级（env 开关 A/B，比 nsys 更可信）

nsys 的 kernel 占比是跨 4 卡求和，和每 token 墙钟不是 1:1；env 开关的 A/B
直接给墙钟差，更可靠。

| 配置 | TPOP | decode | token sha256 |
| --- | ---: | ---: | --- |
| 基线（8K，C=1，graph on） | 13.31 ms | **75.13 tok/s** | `d5fcc5fc…` |
| `FASTLLM_SM70_NVFP4_QPN2=0` | 14.89 ms | 67.16 tok/s | `d5fcc5fc…` |

**QPN2 值 1.58 ms/token，占 decode 的 11.9%**，且 token 流逐位一致（关掉只影响
速度，不影响数值）。这比它在 kernel 占比里的 1.5% 高得多，因为关掉它会回退到
另一条更慢的路径。

### 1.3 已经到位的部分

| 项 | 状态 | 实测 |
| --- | --- | --- |
| CUDA Graph | 已默认开启 | 8K decode 58.57 → 71.89 tok/s（+22.7%） |
| QPN2 原生 NVFP4 | 已接入 Linear | 值 1.58 ms/token |
| NVFP4→FP16 反量化 + TM 回退 | 在位 | 兜底路径 |
| native paged attention（含 E4M3 KV 代码） | 在位 | attention 占比很低 |

## 2. 现状与 1Cat 的差距

| 场景 | FastLLM 本机 | 1Cat | 结论 |
| --- | ---: | ---: | --- |
| 8K C=1 decode | **75.13 tok/s** | ~71 tok/s | **已经领先** |
| 80K C=1 decode | 64.31 tok/s | ~65–71 tok/s | 持平或略低 |
| 80K C=2 聚合 | 108.46 tok/s | 口径不可比 | 瓶颈是 prefill 串行 |

这张表改变了我上一轮的建议。**单请求 decode 已经达到甚至超过 1Cat**，
所以继续移植 kernel 的边际收益很低；而并发路径的瓶颈不在 kernel，
在 prefill 调度（见 `docs/sm70_long_prefill_chunk_plan.md`）。

## 3. 按实测收益排序的移植清单

排序依据是「实测占比 × 该算法能消掉多少」，不是 1Cat 的原始排序。

### 第一优先：TP all-reduce

kernel 占比 **38%–40%**，是最大的一项。1Cat 的 AR 预算 1.587 ms/token
（pack32 push），而我们这里是 NCCL，且 `FastllmCustomAllReduceKernel`
在同一份 trace 里也占 1.2%–1.4%，说明自研路径存在但没赢。

要移植的是 1Cat 的**具体算法**（pack32 push、payload stride 放到 HC 信号之后、
图内安全），而不是打开 FastLLM 现有的 custom AR。注意本机 auto-test 显示
现有 custom AR 在 16 KiB 上只是与 NCCL 打平（18.196 vs 18.606 µs，未达 3% 门槛），
所以这一项必须**先做微基准证明能赢**再接线，否则就是白改。

预期：若能把 AR 成本减半，约等于 decode 的 15%–19%。

### 第二优先：扩大原生 NVFP4 覆盖面

「反量化 + FP16 GEMM」合计 27%–29%（cutlass 128x128 + 反量化 kernel）。
1Cat 用 QPN2/QPN4 全程原生，不做这一步。FastLLM 的 QPN2 形状门是
`M<=32 && N%32==0 && K%64==0`，TP4 下有投影过不了门。

已经量到：GDN-in 的 `N=2608`（`2608 % 32 == 16`）不满足 `N%32==0`。
1Cat 的做法是把 N pad 到 32 的倍数再跑原生 kernel（对应「GDN N=4120→4128
padding」）。把 pad 引到 QPN2 上，让这条投影也走原生路径，是本项最直接的一刀。

预期：先量 GDN-in 一条投影的占比，再决定值不值得做。**做之前必须先测**，
因为 QPN2 的 11.9% 说明原生路径确实有效，但覆盖面到底差多少还没量。

### 第三优先：QPN4 + fused gated SiLU

1Cat 在 NVFP4 gate/up 与 down 上用 QPN4，省 0.86 ms/token（相对它的 TurboMind）。
本地 TurboMind 占 16%–17%，QPN4 能吃其中一部分。

风险：FastLLM 已经用 QPN2 覆盖了同样的形状，所以 QPN2→QPN4 的增益**不等于**
1Cat 的 TurboMind→QPN4 增益，很可能小得多。**必须先做算子级 A/B 再决定**。

### 第四优先：Flash-V100 XQA + E4M3 KV

短上下文下 attention 合计只占 1.5%，所以这一项**对 8K 几乎无收益**。
它的价值在长上下文：80K/180K 时 KV 带宽成为瓶颈，E4M3 KV 直接把 KV 读带宽
减半。移植成本高（整个 `flash-attention-v100/` 目录）。

结论：**先测 80K/180K 的 attention 占比**，占比高才做。不要因为 1Cat 排它
在 P0 就跟着做。

**2026-09-15 已测（80K C=1 decode，TP4，81920 in / 128 out，native paged
attention）。** 原始 trace `/home/nsys/dec80k.nsys-rep`（node 级图回放），
分解脚本 `/home/nsys/analyze_decode.py`（锚 custom AR，128/token，取最后
112 token 窗口，GPU0，占用率 94.8%）：

- attention 占比 **26.2% busy / 24.8% wall**（Split 15.8% + Combine 10.3%），
  过了本节"占比高才做"的门。GEMM 43.4%（QPN2 单 kernel 41.4%），AR 15.2%
  （18.1 µs/次），norm 6.1%。
- **A/B 反转了 E4M3 的位置**：`--kv_cache_dtype fp8_e4m3` 64.95 tok/s，比
  FP16 的 67.08 **慢 3.2%**；`fp4` 两跑 69.72 / 69.88，**快 4.2%**，代价是
  prefill −4.4%（TTFT 41.06 → 42.93 s，prefill 1995 → 1908 tok/s）。三种 KV
  的 token sha256 完全一致（41c7fab5…，greedy）。E4M3 输的原因待查
  （怀疑没走 SM70 GQA D256 专用分支）；FP4 是 `paged-attention-native.cu`
  现成的 native fallback（`docs/fp4-kv-cache.md`）。
- **trace 挖出比 KV 精度更大的杠杆**：`FastllmPagedAttentionCombineGQAKernel`
  是 1 block × 256 线程，97.75 µs/次，16 attention 层 = 1.56 ms/token
  （decode 的 10.5%）——单 block 串行合并 192 个 split 部分和，整卡陪跑。
  并行化它（每 (head, dim-chunk) 一块，线程内在线 softmax 合并 192 个
  split）预计可收回大部分：**约 +10% decode**，与 KV 精度无关。

结论修订：**不移植 flash-attention-v100 的 E4M3 XQA。** 落地顺序改为
① `--kv_cache_dtype fp4`（零代码，+4.2% decode；混合负载要先算 prefill 税的
占比）；② Combine 并行化（+10% 量级，kernel 内改）；③ ①②做完还想挤，再
评估 single-pass 在线 softmax decode kernel（XQA 里真正值得抄的部分，能同时
吃掉 Combine 和 Split 的界）。

### 第五优先：DFlash2/MTP + exact-Philox sampler

这才是 1Cat 从 71 → 80+ tok/s 的那一步（见 1Cat 的 80 tok/s acceptance：
投机 decode + chunked top-20 sampler）。它不是 kernel 优化，是算法层的
投机解码，收益也最大（每轮多接受 token）。

但它是**另一条产品线**：需要 draft 权重、接受率验证、质量门。本路线明确是
no-MTP，所以这一项与当前目标（no-MTP 对齐）冲突。要做得单独立项。

### 明确不移植

DeepSeek/GLM 的 `glm53_*`、`mxfp4_qpn_m1`、`awq_qpn_m1`、MoE
`nvfp4_grouped_decode`、TP8 hierarchical AR、FlashQLA 独立后端。这些属于
别的模型栈，本路线（dense Qwen3.8-27B）用不到。

## 4. 建议的执行顺序

1. **先做 AR 微基准**，用真实 decode 消息尺寸（hidden 5120 × 2 B = 10 KiB，
   以及 20/40/80 KiB 对应 C=2/4/8）证明 1Cat 的算法能赢 NCCL ≥3%。
   赢了再移植；不赢就记一笔，跳过。
   → **已做，不赢。** 1Cat 出厂配置在 10 KiB 是 24.28 µs，NCCL 16.56 µs，慢
   47%；本机 NVLink 全部 inactive，单次跨卡 barrier 就要 9.3 µs。记录见
   `docs/sm70_ar_microbench.md`。auto 探针已改到 10 KiB，TP>=4 从 40 KiB 起硬切 NCCL。
    → **图内 no-end 赢了，引擎还接不上。** CUDA Graph + 独立 dest 上 10 KiB
    是 12.94 µs，graph NCCL 16.05 µs，快 19%。512 次错位回放 4 卡一致。
    引擎调用仍是 in-place，不能只关 `writeAfterBarrier`。下一刀是改图
    （每站点独立目的缓冲），不是改 barrier。见同一份记录 §4。
2. **量 GDN-in 等不过门的投影占比**，决定要不要做 QPN2 的 N-pad。
   → **已做并实测，建议不做。** GDN-in 每 rank 逻辑 N=4120（不是 2608），pad 到
   4128 后确实过 QPN2 的门，kernel 正确性也验过（FP32 oracle + canary + 反向
   对照）。但端到端只有 decode +1.11%、prefill −1.43%：QPN2 在 N=4128 上是
   延迟受限（引擎内 26.66 µs/次），它替换掉的 TurboMind 加 crop 是 32.2 µs/次，
   每次只省 5.5 µs。显存方向与预估相反（大 `--tokens` 下反而释放 0.83 GB，
   交叉点约 67k tokens）。完整实测见 `docs/sm70_qpn_npad_design.md` §8。原本按
   「反量化 + FP16 GEMM 占 27%–29%」给的第二优先，依据是 prefill 污染的占比，
   实测不支持这个位置。
3. 上面两项都做完后，再回头看 QPN4 与 XQA。
4. 并发场景的 prefill 分块与 kernel 移植**正交**，可以并行推进，
   但它是当前并发数字的主瓶颈。
   → **PR-A 已实施并验证（待提交），不再是「未实施」。** 选批分块 + 让位已
   接到 `RunNewMainLoop` 与 `Qwen35MTPLoop` 两条循环：80K C=2 先到那条在
   #1 prefill 期间产出 **40 个 decode token**（改前 ~1），greedy sha256 与
   `FASTLLM_LONG_PREFILL_CHUNK=0` 逐位一致，8K C=4 让位 12/8/4/0。
   PR-B（抬 `GetBatchedPrefillTokenLimit`）实测为死路：round 2+ 线性 KV 非空
   时 `canRunFusedBatchPrefill` 拒绝融合，退回逐行串行，TTFT #0 回退。
   PR-C 重新定性为 kernel 速率门（TTFT #1 是 FLOPs 守恒 163840÷~2000；
   common window 108.46 是 QPN2-on 时代 kernel 速率），无调度缺口。
   详见 `docs/sm70_long_prefill_chunk_plan.md` §5.1/§7/§9。

## 5. 未验证项

- kernel 占比来自含 warmup 的 nsys 抓取。试过 `FASTLLM_SKIP_WARMUP=1` 隔离，
  但它把 decode 从 13.31 拖到 29.08 ms/token（预热缺失），且 nsys 报
  `TargetProfilingFailed`，所以那一路不可用。已用两次不同 workload 的抓取
  互相对照来补偿，但**严格意义上的稳态 decode-only 分布仍未拿到**。
- 80K/180K 的 attention 占比未测，所以 XQA 的优先级还是估计值。
  → **80K 已测（2026-09-15）：attention 26.2% busy；E4M3 KV 实测负收益
  （−3.2%），FP4 KV +4.2%，Combine 单 block 串行是真杠杆（约 +10%）。
  见 §3 第四优先小节。180K 未测。**
- 只测了 C=1 与 C=2。C=4/8/16 全未验证。
- QPN2 的 11.9% 是单点测量（一次 A/B），未做重复取均。
