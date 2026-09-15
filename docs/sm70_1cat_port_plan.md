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
- **trace 挖出比 KV 精度更大的杠杆，已实施并实测。**
  `FastllmPagedAttentionCombineGQAKernel` 旧版 grid=`(batch, numKvHeads)`、
  block=`headDim`；TP4 下每 rank 只有 1 个 kv head，于是**整个 kernel 只有 1 个
  256 线程的 block**，块内还串行 6 个 Q head，每个线程对每个 d 重算 S=192 个
  `exp2f` 因子。改法是把 Q head 提到 grid.y、headDim 切成 grid.z（每块算一次
  因子），数值路径不变（max 扫描与 L 累加仍按原顺序、原值）。
  - 实测 80K C=1：**73.34 / 73.07 tok/s 对旧版 66.66 tok/s，+10.0%**，
    token sha256 `41c7fab5…` 逐位一致，TTFT 不变（41070 对 41148 ms）。
  - **8K C=1 同样 +10.0%**（87.37 对旧版 79.40，同构建 A/B，sha256
    `d5fcc5fc…` 一致）。原因：Combine 扫的永远是 S=192 个槽，S 由固定目标
    384 块推出、与上下文长度无关，所以短上下文下它是更大的固定税。
    TTFT 仍不变（3165 对 3160 ms）。
  - dimChunk 扫过 32/64/128/256，128 最优（13.63 vs 13.64/13.69/13.69），
    默认 128，env `FASTLLM_PAGED_COMBINE_DIM_CHUNK` 可调。
  - Split 切分段数也扫过：`FASTLLM_PAGED_SPLIT_TARGET` 192/384/768 对应
    68.75 / 73.20 / 69.63 tok/s，默认 384 已是最优，不动。
  - 80K C=2 两臂无差（5.82 对 5.79），长 prompt 并发被 prefill 交错主导，
    不是这条 kernel 的赛道。
  - **8K C=4 也是正收益**（33.37 对 31.96 tok/s，+4.4%，sha256
    `5ec42aa4…` 一致，TTFT 不变）。旧版在 C=4 是 4 个 block，新版 48 个。
  - 位级回归：`test/basic/test_cuda_fp4_kv.cu` 加
    `RunCombineParallelParityCase`（FP16/BF16、group=6、headDim=256、
    4097 token，同一输入并行与串行各跑一次逐位比），`bits_diff=0`。
  - 回退开关：`FASTLLM_PAGED_COMBINE_GQA_PARALLEL=0`。
  - 修复后的 trace（`/home/nsys/dec80k_post.sqlite`）：Combine 从 176.0 ms
    降到 33.4 ms（10.3% → 2.2% busy），profiled 口径 62.15 → 69.84 tok/s。
    当前占比：QPN2 46.5%、Split 17.8%、AR 14.3%、其他 10.2%、norm 6.9%。

**KV 精度的推荐在 Combine 修复后反转。** 用新基线（FP16 73.24 tok/s）重测：
`fp4` 69.67（**−4.9%**）、`fp8_e4m3` 65.19（−11.0%），token sha256 仍全一致。
原因是我上一轮的 +4.2% 是拿"没修 Combine 的 FP16"当基线：FP4/FP8 走的是
`useFP4Tiled` / `useSm7xGqaD256Fp8` 两条**本就多 block 的 `CombineExp2*`** 路径，
没吃到这次收益，绝对值没变（69.67 对之前 69.72/69.88），是 FP16 基线涨了。
所以：**FP16 KV 现在是最快的一条，KV 量化不再是 decode 收益选项，只剩省显存。**

结论修订：**不移植 flash-attention-v100 的 E4M3 XQA。** 落地顺序改为
① Combine 并行化（**已落地 +10.0%**）；② 下一个真目标是 Split（17.8%）与
QPN2（46.5%）；③ KV 量化不作为 decode 提速项（FP16 最快），只在需要长上下文
显存时才考虑 fp4。single-pass 在线 softmax decode kernel 仍可作为吃掉
Split/Combine 界的方向，但它现在要跨过 2.2% 的 Combine，性价比更低。

**并发聚合（2026-09-15 实测，修正前一轮的错误结论）。** 融合 batch 路径是
**正常工作的，权重被摊薄**。判据：

- **短 prompt（两请求几乎同时进 decode，`--input_tokens 8`）**：C=1
  **89.22 tok/s**、C=2 **169.97 tok/s**，几乎正好翻倍。C=2 的
  `TPOP avg` 11.77 ms/token 意味着每步 11.77 ms 产出 **2 个 token**。
- **不要用 trace 的"每 128 个 AR 步长"来判融合。** 两条理由：(1) 该指标在
  两个假设下取值相同——串行会把发射数和 AR 数同时翻倍，比值不变，所以它没有
  判别力；(2) 实测的 C=2 trace 里 99% 是**单请求尾段**（10 KiB `grid=2` 的 AR
  有 12666 次，20 KiB `grid=3` 的融合 AR 只有 384 次、共 3 步），拿它当"融合
  步长"是在测错对象。要判融合只能看聚合吞吐，即上一条的短 prompt 对照。

**并发收益取决于"两个 decode 窗口是否真的重叠"，8K 与 80K 是两个相反工况
（2026-09-15 实测）。**

| 工况 | C=1 聚合 | C=2 聚合 | common window 内容 |
|---|---:|---:|---|
| 8K，out=1024 | 68.75 tok/s | **69.20（+0.7%）** | 只有单请求 |
| 80K，out=256 | 61.16 tok/s | **108.46（+77%）** | 510 token = 两个请求 |

- **8K 没有任何并发收益**，因为两个 decode 窗口从不重叠：`#0` prefill 3.18 s
  + decode 1024×12.25 ms，在 **15.71 s 退场**；而 `#1` 的 TTFT 实测 **17.86 s**，
  已经在 `#0` 之后。`out=256` 那跑同样（`#0` 6.33 s 退场，`#1` TTFT 9.21 s）。
  聚合 69.20 对 68.75 落在噪声里。
- **80K 反而重叠**：`#0` prefill 41.5 s，`#1` TTFT 83.1 s，`#0` 的 decode 被
  `#1` 的分块 prefill 拉长到 83 s 之后仍在跑，所以窗口里两个请求都在产 token，
  聚合 108.46（每请求约 54，比单跑 61 略降）。
- **真因：KV 页容量不足导致 prefill 被挂起**（2026-09-15，直接观测）。
  给调度器加了一处 env 门控的每轮打印（`FASTLLM_SCHED_TRACE=1`，打印
  orders/selected/isPrompt/prefillBlocked/canAddPrefill），A/B 对照：

  | `--tokens` | 页数上限 | 被阻塞的轮数 | TTFT max | common window |
  |---|---:|---:|---:|---:|
  | 16384 | 128（限 102） | **27 / 72** | 6703 ms | 85.83 tok/s（31 token=单请求） |
  | 32768 | 256（限 204） | **0 / 45** | 6405 ms | **150.19 tok/s**（58 token=两请求） |

  页数翻倍后阻塞消失、聚合 +75%。所以 8K C=2 没有并发收益是因为
  **KV 页不够两个 8192 prompt 同时在飞**，`qwen3_5.cpp:22713-22720` 的
  `prefillPageCapacityBlocked` 把 `#1` 的 prefill 挂起到 `#0` 退场。这是容量
  配置问题，不是调度策略问题——**扩大 KV 容量（`--tokens`）就能拿到 150 tok/s**。

  （更正记录：我一度用"两个整 prompt = 128 页 > 102"去解释并被自己以
  "该门按 chunk 算"为由撤回；直接 trace 证明机制成立，撤回是错的。教训是
  推断与直证冲突时以直证为准，且不该在拿到直证前反复改结论。）

这解释了为什么 8K C=2 的 `common window` 只报 87：**它量的是单请求**（`#0` 已
退场，窗口里只有 `#1`）。不是 kernel 问题，是 KV 容量配置问题。**可操作结论：
把 `--tokens` 提到 32768 即可让 8K C=2 拿到 150 tok/s。**

**教训：`Batch decode (common window)` 不等价于稳态并发吞吐。** 要看并发
能不能摊薄权重，必须用「两个请求都已进入 decode」的窗口，即短 prompt 或
显式按 TTFT 之后裁剪。这一条也应回填到 `sm70_concurrency_port_plan.md`
的 53.58 tok/s 解释里。

**8K 的并发聚合撞在权重带宽墙上（2026-09-15 实测）。** 同一构建、同一
口径（`Batch decode (common window)`，即所有请求都进 decode 之后的窗口）：

| 批次 | common window 聚合 | 单请求 | TTFT |
|---|---:|---:|---|
| C=1 | **87.0 tok/s** | 87.0 | 3.16 s |
| C=2 | **87.17 tok/s** | 43.6 | 3.18 / 7.77 s |
| C=4 | **86.96 tok/s** | 21.7 | 3.16 / 16.94 s |

三个批次完全持平。原因和 §"QPN2 到内存屋顶"是同一条：batch-1/2/4 的 decode
都要把每卡 5.14 GB 权重每 token 读一遍，HBM 已经满载，加并发只是把同一份
带宽切成更多份。所以短上下文的 decode 聚合**没有可捡的空间**，C=2/C=4 的
`Batch total`（27.7 / 27.8 tok/s）低是 TTFT 串行 prefill 的算术后果，不是
decode 慢。

这条把 8K C=2 那篇文档里的 53.58 tok/s 彻底解释清楚了：它是「~6 s 零产出 +
~3.5 s 全速 C=2 decode」的混合口径，稳态聚合是 87.17。

**QPN2 已到内存屋顶，不是可动项（2026-09-15 实测）。** decode 的 QPN2 是
batch-1 权重流式 GEMM，每 token 必须把每卡权重全读一遍，所以它的下限由 HBM
带宽决定。本机实测屋顶（`/home/arproto/bw_roof.cu`，512 MiB 纯读）：

- V100-SXM2-16GB 实测读带宽 **889 GB/s**（规格 900）。
- 模型 20.56 GB，TP4 每卡 **5.14 GB**，所以 QPN2 硬底 = 5.14/0.889 =
  **5.78 ms/token**。
- 实测 QPN2 **6.26 ms/token，已达屋顶的 92.4%**。就算榨到 100%，token 也只从
  13.65 到 13.18 ms（73.3 → 75.9 tok/s，+3.5%），而且中间还有 tiling 和
  epilogue 的固定开销，实际上拿不满。

所以 QPN2 不再是候选：46% 的占比看着大，但它是带宽墙，不是效率问题。

**180K 已测（2026-09-15）。** 补齐长上下文这一格：

| 上下文 | decode | token | attention 占比 | Split | Combine |
|---|---:|---:|---:|---:|---:|
| 8K C=1 | **87.0 tok/s** | 11.49 ms | 5.7% | 3.4% | 2.4% |
| 80K C=1 | **73.2 tok/s** | 13.65 ms | 20.0% | 17.8% | 2.2% |
| 180K C=1 | **59.1 tok/s** | 16.93 ms | **32.9%** | 31.0% | 1.8% |

180K 的 TTFT 是 118.98 s（prefill 1515 tok/s），token sha256 `b960451a…`。
attention 占比随上下文单调上升，和预测一致：5.7% → 20.0% → 32.9%。

**KV 量化在三个长度上全线为负，而且越长越差。** 180K 交替 A/B（各 2 跑）：
FP16 **59.15 / 58.96**，fp4 **54.19 / 54.24**（**−8.2%**）。80K 是 −4.9%，
8K 是 −1.1%。原因：fp4/fp8 走 `useFP4Tiled` 与 `useSm7xGqaD256Fp8` 两条
tiled 路径，其 dequant 开销随 KV 长度线性增长，而 FP16 走的是 D256 专用
kernel。**KV 量化只剩省显存一个用途，不用于提速。**

**Split 在 80K/180K 都停在 1x-KV 读低的约 1.55 倍。** 用实测 889 GB/s 折算：
80K 的 KV 是 1.34 GB/token（16 层全注意力、每卡 1 个 kv head、FP16），1x 读底
1.51 ms，实测 2.40 ms（**159%**）；180K 的 KV 是 2.95 GB/token，底 3.32 ms，
实测 5.05 ms（**152%**）。两个长度一致，说明是系统性开销而不是噪声。按 2x 读
算底是 3.02（80K）/ 6.64（180K）ms，都高于实测，所以第二遍读有一部分被 L2
吸收，但没吃掉全部。这条是 80K/180K 唯一还没打透的 kernel 级空间（80K 约
0.89 ms/token，180K 约 1.73 ms/token），但最简单的修法（subgroup=6 去重读）
实测 −12%，需要重新设计，不能靠调常量。

**Split 的瓶颈已定：softmax 数学吞吐，不是带宽也不是延迟链。** ncu 在本机
不可用（decode 在 replay graph 里，要关图才能 attach，而关图掉约 20%，人已决定
不采用），所以改用同几何探针（`/home/arproto/split_read_probe.cu`、
`split_softmax_probe.cu`）拆解，80K、S=192、同样的 2 倍读量：

| 探针 | 带宽 | 占屋顶 |
|---|---:|---:|
| 只读（同样的 page cursor 走法） | 786 GB/s | **88%** |
| 只读 + 真实在线 softmax | 456 GB/s | 51% |
| softmax 去掉跨 token rescale 依赖 | 481 GB/s | 54% |
| online softmax + 2 token 展开 | 245 GB/s | 28%（寄存器压力，更差） |

结论：访问形态本身能跑到 88%，所以不是几何问题；去掉串行 rescale 只赚 5%，
所以也不是延迟链；成本在每 token 的 softmax 数学本身（kreg 点积 + shuffle 归约
+ 两次 exp2 + acc 的 fma）。而真 kernel 的 63–66% 比我的复刻（51%）还快，说明
现役 Split 已经接近这个数学吞吐的上限。**要再快只能改算法**（例如降 exp2
次数、把 partial 归约从 shuffle 换成张量核、或换 split 与 group 的切法），
调常量没有空间。

**8K 与 80K 的 decode 预算（同一探针口径，GPU0，占用率 93–94%）。**
AR 的绝对量不随上下文变，所以它在短上下文占比更高（16.4% vs 14.3%）；
QPN2 在 8K 因为分母小、占比反而更高（55.0% vs 46.5%）。

| item | 8K ms/tok | 8K %busy | 80K ms/tok | 80K %busy | 硬底 |
|---|---:|---:|---:|---:|---:|
| QPN2 | 6.28 | 55.0% | 6.26 | 46.5% | 5.78（带宽） |
| all-reduce | 1.88 | 16.4% | 1.80 | 14.3% | ~1.4（探针） |
| attention Split | 0.39 | 3.4% | 2.40 | 17.8% | 1.51（1x KV 读） |
| attention Combine | 0.27 | 2.4% | 0.30 | 2.2% | 已并行化 |
| GEMV fp16 | 0.72 | 6.3% | 0.72 | 5.3% | lm_head，同带宽墙 |
| norm | 0.92 | 8.1% | 0.93 | 6.9% | elementwise |

**Split 的 subgroup 假设被否。** 旧 kernel 每 block 处理 3 个 Q head，完整
group=6 要读两遍 KV（注释在 `:2089-2091`）。试过把 subgroup 提到 6 让 KV 只读
一遍：80K C=1 **64.41 对 73.13 tok/s，慢 12%**。ptxas 实测共享内存
**9280 → 30912 B**（BF16 变体还多 20 B spill），occupancy 掉得比省下的重复读
更多。而且 subgroup=6 已经实现 1x 读却更慢，说明第二遍读本来就被 L2 吸收
（同一 split 的两个 subgroup block 相邻发射、共享同一个 chunk）；按 2x 读算的
底是 3.02 ms，实测 2.40 ms 还低于它。假设否掉，代码已回退，不留开关。

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
   → **已做并实测；「建议不做」的结论已被落地取代。** GDN-in 每 rank 逻辑 N=4120（不是 2608），pad 到
   4128 后确实过 QPN2 的门，kernel 正确性也验过（FP32 oracle + canary + 反向
   对照）。当时端到端只有 decode +1.11%、prefill −1.43%：QPN2 在 N=4128 上是
   延迟受限（引擎内 26.66 µs/次），它替换掉的 TurboMind 加 crop 是 32.2 µs/次，
   每次只省 5.5 µs。其后 npad 侧车随 landing 方案（A 单元）进树并按缺省开启，
   最新普查证实 decode 256 条 NVFP4 投影已**全部**走 QPN2（GDN-in 在
   grid=129，25.64 µs/次；窗口里无 TurboMind 回退、无 crop kernel）。
   显存方向与预估相反（大 `--tokens` 下反而释放 0.83 GB，交叉点约 67k tokens）。
   完整实测见 `docs/sm70_qpn_npad_design.md` §8。剩余杠杆（per-shape 几何、
   lm_head）移入 `docs/sm70_nvfp4_coverage_plan.md`。原本按
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
  → **C=4 已测（2026-09-15，8K）：combine 并行化 +4.4%，common window
  86.96 tok/s。C=8/16 仍未验证。**
- 180K 的 attention 占比未测。
  → **180K 已测（2026-09-15）：59.1 tok/s，attention 32.9% busy，KV 量化
  −8.2% 且随长度变差。见 §3。**
- QPN2 的 11.9% 是单点测量（一次 A/B），未做重复取均。
- **工具面：ncu 在本机这条 workload 上默认 kernel replay 崩，`--replay-mode
  application` 是唯一跑通的模式但拿不到硬件计数器。** 见 §5.1。

### 5.1 ncu 归因的可用面（2026-09-15，本机实测矩阵）

背景：`.audit/sm70-longctx-kv.tsv` 曾把整条路线判成「ncu cannot profile this
workload」。该记录的表述有两处不准确，已在该文件追加 CORRECTION 行：
`--cuda-graph-trace` 是 **nsys** 的开关（ncu 报 `unrecognised option`，exit 1），
且 CUDA Graph 与 kernel replay 不是原理性冲突（`ncu --graph-profiling` 默认
`node`）。但**结论本身成立**，理由被本轮实测改写了，见下表。

**实测矩阵**（本机 4×V100，`--input_tokens 8192 --output_tokens 4 --batch 1`，
目标一律是 `--kernel-name regex:FastllmCudaNVFP4Block162HalfKernel`，
`--launch-skip 20 --launch-count 2 --set basic`）：

| # | graph | ncu replay 模式 | 结果 |
| --- | --- | --- | --- |
| T1 | on | 不加 ncu（对照） | 正常，exit 0，sha256 `85cb21de…` |
| T2 | on | kernel replay（默认） | **exit 11（SIGSEGV）**，崩在 warmup 后，report 未写出 |
| T3 | off | kernel replay（默认） | **exit 9**，`UnknownError`，死在 kernel replay 的 save-and-restore 步骤 |
| T4 | on | `--replay-mode application` | **不崩**，权重驻留 16 GB/card，进稳态（随后被共租户挤掉而 kill） |

**T2 与原记录的崩溃不是同一个信号，所以它不是逐字复现。** 原记录是
`ERROR app returned 6`（SIGABRT，`double free or corruption`）；T2 是
exit 11（SIGSEGV），dmesg 显示 fault 落在 **`libcuda-injection.so`**（ncu 自己的
注入库，12:37:53 那条），不是 glibc 的堆检查。两者同属「graph=on 时 ncu kernel
replay 介入后崩」，但崩溃点不同。**结论按「同机制、不同点位」记，不按复现记。**

两条被实测推翻的推断（都出自我上一轮）：**「prefill 侧零代价，因为 prefill
不进图」不成立**。T2 与 T3 的目标核是 prefill-only 的反量化核，filter 收窄到
它一个，照样在 graph=on 时 exit 11。原因是 decode 的图在 **warmup 阶段**就
capture 了，跟 profile 哪个核无关；ncu 的 kernel replay 一旦介入，就打在
capture 期的指针记账上。

所以正确的说法是：

- **graph=on + 默认 kernel replay：崩（exit 11，SIGSEGV）。** 无论目标核是
  decode 还是 prefill，只要分母里有 capture 就崩。这是 `.audit` 那条记录的真
  实机制。
- **graph=off + 默认 kernel replay：不 SIGSEGV，但失败（exit 9，`UnknownError`）。**
  它死在 ncu 为 kernel replay 做 save-and-restore 那一步（log 只有
  `Backing up device or system memory to file` 一句 `WARNING` 紧接
  `==ERROR== UnknownError`）。具体耗尽的是显存、系统内存还是磁盘，本次证据
  不足以判定；能说的是它没有产生 report，且发生在权重加载之后。
- **graph=on + `--replay-mode application`：不崩，能进稳态，但拿不到硬件计数
  器。** 它不做 kernel 重放，所以既不触发 capture 冲突，也不走那步
  save-and-restore。本次只观察到它加载完权重、进入稳态（随后被共租户挤掉而
  kill，没等到 report）；按该模式的定义它只给 launch / duration 级数据，
  SM/DRAM 段与指令统计拿不到，而算子级归因要的正是那些。**结论：对本用途无
  价值。**

**结论：ncu 在本机这条 workload 上不可用于算子级归因。** 与 `.audit` 原记录
的不同只在理由：不是「图与 kernel replay 天然冲突」，而是「graph=on 时 capture
× kernel replay 崩」与「replay 的 save-and-restore 步骤在本机失败」这两条，且
两条都被实测钉住。想测就得改图的 capture 结构，或换一台 replay 开销不撞墙的
机器，都不在当前范围内。

**回退：nsys node 级 trace 仍是本机唯一可用的归因手段**，这也是原记录的
决定，保留。

**顺带产出（纯 CPU，不依赖 ncu）**：现有 80K trace 的核时按 `graphId` 拆开，
prefill 侧的账可以直接读出来。

**prefill 的核时账（`/home/nsys/dec80k_post.sqlite`，`graphId` NULL=eager，
非空=图内；CPU 查询）：**

| 侧 | 核时 | 占比 |
| --- | ---: | ---: |
| eager（prefill 为主，含少量未进图的 decode 步） | 174.5 s | **95.9%** |
| 图内（decode） | 7.5 s | 4.1% |

（eager 桶不是纯 prefill：其中 11,264 个 `nvfp4_qpn2_sm70_kernel`、约 0.3 s 是
未进图的 decode 步。prefill 的 M=2048 GEMM 占绝对多数，所以 95.9% 是 prefill
的可靠近似，但不是「纯 prefill」的精确值。）

prefill 的 eager 侧拆开（排除 NCCL 37% 后的纯计算 110.7 s）：

| 族 | 核 | 时间 | 占计算 |
| --- | --- | ---: | ---: |
| `cutlass::Kernel2<cutlass_70_tensorop_h884gemm_128x128_tn_align8>` | 45,056 | 49.4 s | 44.6% |
| volta FP16 GEMM（`volta_s884gemm_fp16_64x64_ldg8_nn`、`volta_h884gemm_256x128_ldg8_tn`） | 1,763,904 | 33.3 s | 30.1% |
| attention（`FastllmPagedCublasSoftmaxWithCausalMask`、`FastllmPagedCublasAttnBlockUpdateFloat`） | 630,086 | 13.1 s | 11.8% |
| **`FastllmCudaNVFP4Block162HalfKernel`（NVFP4→FP16 反量化）** | 45,056 | **7.4 s** | 6.7% |
| 其余（norm/swiglu/conv1d/GDN） | | 7.6 s | 6.9% |

两个可直接读出来的结论：

1. **反量化核 7.37 s 全在 prefill，图内为 0。** 因为 QPN2 是 decode-only
   （`fastllm-linear-fp8.cu:3679`：`QPN2 is decode-only (M <= 32)`；M>32 落到
   `FastllmCudaNVFP4Block162HalfKernel` + cuBLAS）。这与 §1.2「反量化占
   decode 3.8%–4.4%」不矛盾：那是 decode 窗口的口径，这里是全 trace 口径，
   prefill 的 M=2048 GEMM 把权重压过一遍。**prefill 的 6.7% 是唯一一处
   「QPN2 覆盖面」还能产生 prefill 收益的地方，但它属于「扩 M」不是
   「扩 N」，与 §2 的 QPN4 计划正交。**
2. **prefill 计算 74.7% 在两族 GEMM 上**（cutlass tensorop 44.6% + volta FP16
   30.1%）。这两族的占用可以用 trace 里的 `registersPerThread` / block / smem
   静态算出来（纯 CPU，不需要 ncu）：

   | 核 | 完整 grid | block | reg | smem | 上限 blocks/SM | 占用 |
   | --- | --- | ---: | ---: | ---: | ---: | ---: |
   | `cutlass::Kernel2<…h884gemm_128x128…>` | (128,5,1) | 128 | 168 | 32768 | 3（寄存器限） | 18.8% |
   | `volta_h884gemm_256x128_ldg8_tn` | (8,16,1) | 256 | 138 | 49664 | 1（smem 限） | 12.5% |
   | `volta_s884gemm_fp16_64x64_ldg8_nn` | (4,32,2) | 128 | 106 | 16640 | 4（smem 限） | 25.0% |

   这些是 V100 上大 tile GEMM 的**常规**数字（128×128 tile 天然寄存器重），
   所以**不能据此判定有优化机会**。它们到底 stall 在内存还是发射，要硬件计数器，
   而 ncu 在本机不可用（§5.1）。**这一项到此为止，除非换工具。**
   但它是 TTFT #1 守恒项（163840 ÷ ~2000 tok/s ≈ 82 s）里没被解释的部分，
   值得记着。

**取证项状态：被工具挡住，不是被 GPU 挡住。**

- [x] P-N1（反量化核 M=2048 发射几何）与 P-N2（
  `FastllmPagedCublasSoftmaxWithCausalMask` 的 causal-mask 分支）**都无法用
  ncu 取证**：本机两种 ncu 模式分别崩（exit 11）或 OOM 掉 ncu 自己（exit 9），
  能跑通的 `--replay-mode application` 恰好不产出硬件计数器。见 §5.1 的 T1–T4
  矩阵。两个问题仍然开放，但要用别的工具或别的卡，先不占预算。
- [x] P-N3（前置确认）**本轮执行过并踩到过**：12:31–12:40 之间有一次共租户占用
  （`nsys … --input_tokens 180224 --kv_cache_dtype fp4`，每卡 ~10.5 GB、100%），
  期间我的 run 被挤到每卡 5.5 GB 且无进展。按时等、不挤的原则，已 kill 掉自己
  的 run，共租户的进程没动。此后每次跑前照旧确认四卡 <1GB。

**明确不做（graph off 不是可选项，已由操作决策关闭）：** 操作指令
「不要考虑关闭 Graph，负性能」。实测支持该判断：8K C=1 关图 64.28 对开图
87.34 tok/s（−26.4%），8K C=2 关图 130.50 对 162.05（−19.5%），两对的
sha256 都逐字相同（`02c702ca`、`5bfaac89`），即关图只损失吞吐、不换任何正确性。
所以「关图以便 ncu 测 decode 核」**不构成一个可讨论的选项**，本节 decode 侧
的结论因此是终局：本机不做 decode 核级归因。

**也不做：** 在小显存卡上硬试 ncu kernel replay（save-and-restore 撞墙）。

## 6. 各杠杆判定汇总（2026-09-15）

短上下文 decode 已经钉死。这一轮把所有能动的项都判完了：

| 项 | 占比（8K/80K/180K） | 判定 | 依据 |
|---|---|---|---|
| QPN2 | 55.0% / 46.5% | **封顶，不动** | 实测 92.4% 于 889 GB/s 的 HBM 屋顶 |
| lm_head GEMV | 6.3% / 5.3% | 同带宽墙，**已达 99.8%** | 0.64 GB/卡分片 ÷ 889 GB/s = 0.72 ms 是底，实测 0.719 |
| AR | 16.4% / 14.3% | **唯一剩下的杠杆** | push 上限 +10.8% / +9.1%，中心 +4–6% |
| attention Split | 3.4% / 17.8% / 31.0% | **已定性：softmax 数学吞吐**，现役已近上限，再快要改算法 | subgroup=6 −12%；探针：只读 88% 屋顶、加 softmax 51%、去 rescale 仅 54% |
| attention Combine | 2.4% / 2.2% | **已落地** | +10% @ 8K/80K C=1，+4.4% @ 8K C=4 |
| KV 量化 | n/a | 不作为提速项 | fp4 −4.9%、fp8 −11.0%（相对 FP16 基线） |
| 并发聚合 | n/a | **收益取决于 decode 窗口是否重叠：8K +0.7%，80K +77%** | 8K out=1024 69.20 对 68.75；80K out=256 108.46 对 61.16 |
| 长上下文 180K | attention 32.9% | KV 量化更差 | fp16 59.1 vs fp4 54.2 tok/s |
| norm/swiglu/conv1d | 8.1%/1.5%/1.7% | 延迟地板 | 单 block 4.7/2.7/3.7 µs，不随上下文变 |

**结论。** 单请求 decode 的每一项都有实测把它判死（带宽墙、延迟地板、或已被
Combine +10% 吃掉）；kernel 侧唯一还有量级空间的仍是 **AR push（+4–6%）**，
且随上下文变长而变小（8K 15.8% → 80K 13.2%）。

并发侧分两层。kernel 侧无缺陷：融合 batch 正常、权重正常摊薄（短 prompt C=2
= 170 tok/s ≈ 2×C=1）。调度侧的 8K 缺口已定性为 **KV 页容量不足**：
`--tokens` 16384（128 页）时 prefill 被挂起 27/72 轮、聚合 85.83 tok/s；
提到 32768（256 页）后阻塞为 0、聚合 **150.19 tok/s（+75%）**。
这是配置项而非代码改动，可直接采用。
