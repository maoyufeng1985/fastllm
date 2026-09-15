# SM70 路线整合：功能边界与待优化方向

日期：2026-09-15
范围：Qwen3.8-27B-**QUASAR**-NVFP4 / SM70 / 4×V100-SXM2-16GB（PCIe fabric，无 NVLink）/ TP4 / no-MTP
本文性质：**入口文档**。把 `docs/sm70_*.md` 共 14 篇（约 4800 行）的结论收敛成一份
现状与待办；细节、证据与反例仍在各专项文档里，本文只做索引与判定汇总。

**证据等级**（本文每条结论都能归到四类之一，便于判断该不该被推翻）：

- **[码]** 在本仓库源码里核实过（缺省值、形状门、编译期常量、调用点）。
- **[测]** 源文档记录的实测数字，运行时可复现；本文只转引，不新增测量。
- **[案]** 只是方案/设计，尚未落地。
- **[废]** 已被后续实测推翻或更正，保留标记防止复发。

**当前工作区基线**：`master` = `7fc5e06e`（本轮提交的 4 个 sm70 文档/工具提交），
比 `maoyufeng/master` 领先 4 个、比 `origin/master` 领先 14 / 落后 32。工作区除
`.audit/` 外另有第二作者的在制品（诊断开关、`sm70_tp4_push_ar_plan.md` 等），
引用本文件时以 `git status` 为准。

## 0. 一句话

单请求 decode 已经**钉死**：8K 87.0 / 80K 73.5 / 180K 59.1 tok/s，余下的项都被
实测判成带宽墙、延迟地板或已落地的收益（§3）。**并发是目前唯一的活口**，而它
的瓶颈不在 kernel 在调度：8K C=2 的缺口是 **KV 页容量不足**（配置项，`--tokens`
翻倍即 +75%），80K C=2 的缺口是 **长 prompt 整段放行**（已落地选批分块，公平性
改善，聚合代价 −6.5 tok/s 换 `Total time` +0.7%）。

## 1. 冻结合同（改这条路线前先确认没变）

| 项 | 值 |
|---|---|
| 模型 | `/data/models/Qwen3.8-27B-QUASAR-NVFP4`（dense，无 experts） |
| hidden / intermediate | 5120 / 17408（gated，TP4 本地 K5120×N8704） |
| 层数 | 64（`full_attention_interval = 4`，其余 48 层是 GDN 线性层） |
| head_dim / GQA | 256 / 24:4 |
| 量化 | `nvfp4-pack-quantized`，group 16，E4M3 scale，**ignore lm_head** |
| 硬件 | 4×V100-SXM2-**16GB**，NVLink 全 inactive（`nvidia-smi nvlink -s`），
`nvidia-smi topo -m` 全 PIX；单次跨卡 barrier ≈ 9.3 µs |
| 并行 / 精度 | TP4，FP16 KV（FP4/E4M3 见 §3.4），CUDA Graph decode |
| 调度路径 | 生产 auto `FASTLLM_GPU_TOKEN_HANDOFF=1` → `Qwen35MTPLoop`；否则 `RunNewMainLoop` |
| 实测 HBM 读带宽 | 889 GB/s（`/home/arproto/bw_roof.cu`，512 MiB 纯读；规格 900） |
| 模型权重 | 20.56 GB / TP4 = **5.14 GB 每卡每 token 必读** |

**1Cat 对照的坑**：1Cat 的 71 tok/s / 14 ms 是**短到中上下文**的数字，FastLLM 的
50.3 tok/s 是 **180K** 的数字。两者不能直接比，否则会把「长上下文 KV 带宽」误判成
「kernel 缺口」（`sm70_qwen38nvfp4_1cat_gap_audit.md` §5）。

## 2. 功能边界：现在到底有什么

### 2.1 已落地且缺省就开

| 能力 | 落地提交 | 实测 |
|---|---|---|
| SM70 CUDA Graph 默认开（`util.py` 门槛 `> 75` → `≥ 70`） | `ffc079e4` | 8K C=1 +22.7%、80K C=1 +25.7%、80K C=2 common window +11.5%，greedy sha256 不变 |
| QPN2 原生 NVFP4 GEMM，接 Linear 分发（QPN2 → TurboMind → Marlin → GEMV/cuBLAS） | `495e9d73`，GDN-in N-pad 侧车 `08cfe5c4` | 覆盖 256/256 条投影（GDN-in 走 grid=129、25.64 µs/次，窗口内无 TurboMind 回退、无 crop）。**当前二进制勾掉 Combine 的回退 A/B**：8K C=1 +23.7%、8K C=2 +62.6%、80K C=1 +22.4%、80K C=2 +48.9%、8K C=4 +101.4%（跨源）；8K/80K prefill 持平 |
| TP decode all-reduce：独立 dest + 去 end barrier + `auto` 分档（10 KiB small / ≥40 KiB 硬切 NCCL） | `fa0b6030` | 8K C=1 +9.6%、80K C=1 +8.1%、8K C=2 +7.9%，sha256 逐位一致 |
| GQA D256 Combine 并行化（Q head → `grid.y`、headDim → `grid.z`，dimChunk 128） | `debbf431` | 8K/80K C=1 各 +10.0%，8K C=4 +4.4%；位级回归 `bits_diff=0` |
| 长 prefill 选批分块 + 让位（PR-A，`FASTLLM_LONG_PREFILL_CHUNK` 默认 on） | `9647b3bd` | 80K C=2 期间先到者产出 40 token（改前 ~1）；8K C=4 让位 12/8/4/0 |
| benchmark `Batch decode (common window)` 口径 + `_token_stream_hash` | `1c6da2b0` | 见 §5「口径」 |
| MoE cache 行数上限 9 → 16 | `6d08633e` | 只放宽接受行数，无额外显存 |

组合收益（三单元全开、当前二进制同源）：80K C=1 **73.51**（门 61.16）、
80K C=2 common window **115.39**（门 108.46，且保留 40 token 让位）、
8K C=4 **270.57**（旧门 129.27 是 QPN2-off 时代的跨源值）、
80K C=2 TTFT #1 82.95 s（门 83.09）。
出处 `sm70_npad_ar_chunk_landing_plan.md` Appendix C。

### 2.2 已编好但**没接线**（别误当能力）

| 项 | 状态 | 为什么没接 |
|---|---|---|
| QPN8 FP8 kernel（`qpn8_fp8.cu`） | 编译在、`sm70QpnFp8Regression` 8/8 过、`src/` 下**零调用** | 本 checkpoint 纯 NVFP4，QPN8 需要一整套 channel-FP8 权重契约；1Cat 自己的 online QPN8 因改变 token 轨迹被验收显式关闭 |
| TP4 push all-reduce（`FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4`） | 只有方案与探针，未写引擎代码 | 2026-09-15 人决策**暂缓**：中心收益 +3%–4% 不值得当前投入。探针保留可随时执行 |
| `FASTLLM_PREFILL_PAGE_POLICY` | 只有方案（C2 TTFT 方案 §4 的 U2） | 需先有 U0/U1 的账本证据 |
| lm_head NVFP4 | 只有立项卡（U4） | 改变 logits，sha256 门必然失败，需换质量门（GSM8K + ppl），**明确不进默认范围** |

### 2.3 所有相关 env 开关与缺省

| env | 缺省 | 作用 |
|---|---|---|
| `FASTLLM_SM70` / `FASTLLM_SM70_QPN` / `FASTLLM_SM70_NVFP4_QPN2` | 全 on（`EnvEnabled` 未设即真） | QPN2 三层门，任一置 0 关闭 |
| `FASTLLM_SM70_FP8_QPN8` | on（但无调用方） | QPN8 门 |
| `FASTLLM_CUDA_GRAPH` / `FASTLLM_QWEN35_SM70_CUDA_GRAPH` | on | 后者只在**全部**选中设备 <7.5 时生效（混合组不受影响） |
| `FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH` | 见 `util.py` | 图预捕获的 batch 形状列表 |
| `FASTLLM_CUDA_CUSTOM_ALLREDUCE` | `auto` | `auto` 走 auto-test 分档；`0` 强制 NCCL；TP≥4 且 ≥40 KiB 硬切 NCCL |
| `FASTLLM_PAGED_COMBINE_GQA_PARALLEL` | on | Combine 并行化回退开关 |
| `FASTLLM_PAGED_COMBINE_DIM_CHUNK` | 128 | 扫过 32/64/128/256，128 最优 |
| `FASTLLM_PAGED_SPLIT_TARGET` | 384 | 扫过 192/384/768，384 最优 |
| `FASTLLM_LONG_PREFILL_CHUNK` | on | 选批分块 + 让位回退开关（`=0` 逐位对照锚） |
| `FASTLLM_SCHED_TRACE` | off | 每轮打印调度快照（`qwen3_5.cpp:22888`），根因取证用 |
| `FASTLLM_GPU_TOKEN_HANDOFF` | 生产 auto 开 | 决定走 `Qwen35MTPLoop` 还是 `RunNewMainLoop` |
| `FASTLLM_PAGED_CUBLAS_CHUNK` | 8192（80K 需 2048） | 8192 在 16GB 卡上会把 QK workspace 顶到 ~1.3 GB 并挂死 |

（`FASTLLM_CUDA_MOE_CACHE_MAX_BATCH` 是 `fastllm-cuda.cuh:1601` 的**编译期常量** 16，
不是 env；它同时是 MoE cache 的行数门与 `qwen4_exp.cpp:7941` 的图条件。）

### 2.4 明确划出范围（不做，且理由已被实测钉住）

1Cat pack32 AR、Flash-V100 XQA / E4M3 XQA、KV 量化提速、QPN8 接线、QPN4 扩覆盖、
fused AllReduce+RMSNorm、CUTLASS SM70 prefill、DFlash2/MTP5、PushAdd 扩 TP4、
PairAdd 合 attn+MLP、MoE grouped / 其他模型栈（DeepSeek、GLM、AWQ-QPN、MXFP4）、
关 CUDA Graph（操作决策：关图只掉 19%–26% 吞吐，不换任何正确性）、
ncu 做 decode 核级归因（本机两种 replay 模式分别崩 / 拿不到计数器）。

## 3. 现状基线（当前二进制，同源）

### 3.1 decode 稳态

| 工况 | decode | token | attention 占比（Split） | 备注 |
|---|---:|---:|---:|---|
| 8K C=1 | **87.0 tok/s** | 11.49 ms | 5.7%（3.4%） | Combine 并行化 +10% 已含在基线内 |
| 80K C=1 | **73.2 tok/s** | 13.65 ms | 20.0%（17.8%） | |
| 180K C=1 | **59.1 tok/s** | 16.93 ms | 32.9%（31.0%） | TTFT 118.98 s（prefill 1515 tok/s） |
| 8K C=2 common window | **161.72** | — | — | AR off 149.86 |
| 8K C=4 common window | **270.57** | — | — | 让位 12/8/4/0 保留；对照 129.27 是 QPN2-off 时代的旧值（跨源） |
| 80K C=2 common window | **115.39** | — | — | 保留 40 token 让位；零让位对照 121.90 |
| 短 prompt 8K C=2（两请求都在 decode） | 169.97 | 11.77 ms/步 | — | C=1 89.22，几乎正好翻倍 → 权重确实被摊薄 |

### 3.2 8K / 80K decode 预算（同一探针口径，GPU0，占用率 93–94%）

| item | 8K ms/tok | 8K %busy | 80K ms/tok | 80K %busy | 硬底 |
|---|---:|---:|---:|---:|---:|
| QPN2 | 6.28 | 55.0% | 6.26 | 46.5% | 5.78（带宽，已达 92.4%） |
| all-reduce | 1.88 | 16.4% | 1.80 | 14.3% | ~1.4（no-end 探针）/ 0.68（push 探针） |
| attention Split | 0.39 | 3.4% | 2.40 | 17.8% | 1.51（1x KV 读，实测 152%–159%） |
| attention Combine | 0.27 | 2.4% | 0.30 | 2.2% | 已并行化 |
| GEMV fp16（lm_head） | 0.72 | 6.3% | 0.72 | 5.3% | 0.719（已达 99.8%） |
| norm | 0.92 | 8.1% | 0.93 | 6.9% | elementwise |

### 3.3 并发基线

| 工况 | C=1 | C=2 | common window 里是什么 |
|---|---:|---:|---|
| 8K out=1024 | 68.75 | **69.20（+0.7%）** | 只有单请求（#0 已在 #1 TTFT 前退场） |
| 80K out=256 | 61.16 | **108.46（+77%）** | 两请求都在（510 token） |

**8K 并发没有收益的真因是 KV 页容量**（`FASTLLM_SCHED_TRACE` 直证）：
`--tokens 16384`（128 页、限 102）时 prefill 被挂起 **27/72 轮**、聚合 85.83；
提到 32768（256 页、限 204）后阻塞 **0/45**、聚合 **150.19（+75%）**。
这是**配置项，不是代码改动**。

**8K 并发聚合的"持平"要带条件读**：C=1/C=2/C=4 的 common window 分别 87.0 / 87.17 /
86.96，看着完全持平；但这张表是在 `--tokens 16384` 下测的，窗口里只有单请求
（`sm70_c2_ttft_overlap_plan.md:26-30`）。同文档 V5 首次给出真双请求窗口：
每请求 ~80 tok/s、并发步长 **≈13.3 ms ≈ 1.16× C=1**，即**权重是被摊薄的**——
这与旧的"不摊薄 2.00×、每步 22.94 ms"**直接矛盾，二者必有一错**。
窗口口径上"C=2 单请求 43.6"是把 87.17 对半折算的假设值，不是实测。
所以「8K 撞权重带宽墙」目前是**待裁决**，不是结论；裁决实验见 T1 的 U0。

### 3.4 长上下文与 KV 精度

| 上下文 | FP16 | fp4 | fp8_e4m3 |
|---|---:|---:|---:|
| 8K C=1 | 87.0（基线） | −1.1% | — |
| 80K C=1 | 73.24 | −4.9% | −11.0% |
| 180K C=1 | 59.15 / 58.96 | **−8.2%** | — |

三种 KV 的 greedy token sha256 完全一致。**KV 量化只剩省显存一个用途**：
fp4/fp8 走 `useFP4Tiled` / `useSm7xGqaD256Fp8` 两条 tiled 路径，dequant 开销随 KV
长度线性增长，而 FP16 走 D256 专用 kernel。16GB 卡的余量按配置走：180K 用
`--tokens 230400`（1800 页）加载后单卡约 14.0 GB / 16 GB（`sm70_concurrency_port_plan.md` §16.1）。
（**注意**：这条结论在 Combine 修复前后**反转**过一次——修复前 fp4 看着 +4.2%，
是因为拿"没修 Combine 的 FP16"当基线。引用旧数字会得出相反结论。）

### 3.5 口径（读任何数字之前先看这段）

- **`Batch decode after TTFT` 在 batch>1 时不代表 decode 速率**：窗口从**第一个**
  TTFT 起算，把另一条请求仍在跑的 prefill 记进 decode。8K C=2 的 53.58、80K C=2 的
  6.72 / 11 都是这个口径的产物。
- **并发只看 `Batch decode (common window)`**（从**最后一个** TTFT 起算）。
- **sha256 只在同 `--output_tokens` 下可比**：`_token_stream_hash` 把全部生成 token
  一起哈希。曾经把 out=64 与 out=256 的 sha 当成「跨二进制漂移」，那是口径错。
- **QPN2 的 kernel 占比只有 clean decode 窗口能读**：全 trace 聚合会被 prefill 的
  M=2048 压过（早期"QPN2 只占 1.4%"的表就是这么来的，已作废）。
- 生产路径走 `Qwen35MTPLoop`（handoff），不是 `RunNewMainLoop`。

## 4. 待优化方向（按证据强度与收益排序）

### T1 —— 8K C=2 的 KV 页容量（配置，零代码，最高确定性）

`--tokens 16384 → 32768`：common window 85.83 → **150.19（+75%）**，Batch total +4%，
代价 +269 MB/卡（卡上还有 ~1.4 GB 余量），`promptLimit` 升到 26112。
**已实测、可直接采用**，缺的只是把 8K 档基准命令与文档口径一起改掉
（`sm70_c2_ttft_overlap_plan.md` §4 的 G1）。

### T2 —— 页守卫策略（把 T1 的收益在 16384 池上拿到）

现在 `qwen3_5.cpp:22713-22720` 的 `prefillPageCapacityBlocked` 粒度是**整请求等**，
而不是「把 chunk 切小到当下装得下」。三个实施单元（`sm70_c2_ttft_overlap_plan.md` §4）：

- **U0 复核实验（先做，一锤定音）**：裁决并发 decode 步长是 ~1.16×（权重分摊）
  还是 ~2.00×；这一条决定「C>1 权重不摊销 +37%」这个旧结论要不要推翻。
- **U1 守卫计数器**：阻塞时打印 `need/free/limit/请求长度/chunk` 五元组，
  确认缺的是 chunk 级还是整 prompt 级页需求。
- **U2 守卫最小修复**：`(a)` 预留只计下一 chunk + decode 增长页；或
  `(b)` chunk 尺寸自适应收缩到 `free - decodeReserve`。env
  `FASTLLM_PREFILL_PAGE_POLICY=strict|grow`（缺省 strict）。
  **门**：16384 池上阻塞 0/72、sha256 与 32768 跑逐位一致、80K/180K 不回退不 OOM。
- **U3 让位粒度**：PR-A 的让位从每 chunk（2048）改到每 N token（512/256）。
  **门**：8K C=2 Batch total 升、#1 TTFT 劣化在定点内。

### T3 —— QPN2 per-shape 几何（唯一的 kernel 级余量）

现状：256 条投影已**全部**走 `nvfp4_qpn2_sm70_kernel`（`sm70_nvfp4_coverage_plan.md` §1）。
按形状拆（80K post-combine trace、device 0、112 token 窗口）：

| 形状组 | 本地 K×N | grid | 条/token | µs/次 | GB/s（%roof 900） |
|---|---|---:|---:|---:|---:|
| MLP gate/up | 5120×8704 | 272 | 64 | 37.34 | 671（74.6%） |
| N=5120 组（down / GDN-out / attn-O） | 5120/4352/1536×5120 | 160 | 128 | 17.64 | 480（53.3%） |
| GDN-in（侧车 4128） | 5120×4128 | 129 | 48 | 25.64 | 463（51.5%） |
| attn QKV | 5120×3584 | 112 | 16 | 24.49 | 421（46.8%） |
| 合计 | | | 256 | 6.27 ms/token | 546（60.7%） |

共性是 **CTA 数不足、wave 尾差**（80 SM 上 grid 112/129/160 = 1.4/1.61/2.0 waves），
而 gate/up 的 grid 272（3.4 waves）已打到 671 GB/s——这就是「kernel 本体没问题、
差在发射几何」的证据。roof 账：3.424 GB/rank/token（0.5625 系数已含 E4M3 scale），
对 85% roof（765 GB/s）的差 = **1.79 ms/token（12.5%）**，是全组理论上限。

- **L1**：`ChooseSplitK` 按形状分发（N=4128/3584 用 split 16，CTA 翻倍）、
  N-tile 16 双 tile。上限 ~1.6–1.8 ms/token；**立项门 ≥+3%（~0.45 ms），目标 ≥+6%**。
- **L2（附带）**：QPN2 fused gated-SiLU epilogue，折叠 `FastllmSwigluKernel`（164 µs）
  与 `RMSNormSiluMulHalf128Exact`（167 µs），上限 ~0.33 ms。
- **L3（单独立项）**：lm_head NVFP4，上限 ~0.31 ms，但**换质量门**，不进默认范围。
- **顺序**：U1 per-shape 探针（`bw_roof.cu` / `qpn2_gdn.cu` 变体矩阵）→ 任一形状
  ≥600 GB/s 才进 U2 接线 → U3 长上下文与并发复测。开关 `FASTLLM_SM70_QPN2_GEOM`
  是规划名，**当前树里不存在**（`grep` 零命中；U2 落地时才加，缺省 auto=现役）。

### T4 —— attention Split（长上下文唯一大项）

180K 时 Split 占 31.0%（5.05 ms/token，1x-KV 读底的 152%）。已定性：**瓶颈是
softmax 数学吞吐，不是带宽也不是延迟链**（同几何探针：只读 88% roof、加真实 online
softmax 掉到 51%、去掉跨 token rescale 只回 5%）。现役 kernel 的 63–66% 已经比复刻
探针快，**调常量没有空间**（subgroup=6 试过 −12%，shared 9280→30912 B，已回退）。
要再快只能改算法：降 exp2 次数、partial 归约从 shuffle 换张量核、或换 split/group 切法。
80K 约 0.89 ms、180K 约 1.73 ms 是账面余量。

### T5 —— TP4 push all-reduce（暂缓，条件重启）

探针：push 变体背靠背 6.3 µs vs 现役 14.0 µs（10 KiB）；引擎侧 AR 桶 1.876 ms/token
（8K，device-0 mean 14.65 µs，16.4%）、1.925 ms（80K，mean 15.04 µs，14.1%）。
上限 +11.7%（8K）/ +10.1%（80K），**中心估计 +4%–6%**（决策行 +3%–4%）。
历史兑现率参照：skip-end 省 640 µs 只兑现 150 µs（23%）。
**重启条件**：decode 墙钟重新成为主要目标，或其他更高收益项做完后仍需挤 AR。
U1–U4 与判定门（中位 decode ≥+1.5%、sha256 `d5fcc5fc…`、
`customAllReduceRegression` PASS）已写在 `sm70_tp4_push_ar_plan.md`。

### T6 —— 收 `NcclForceSync` 与 40–512 KiB 档

SM70 仍开 `NcclForceSync`（`basellm.cpp` 里 `RuntimeArch() >= 75` 才关），理由是
长 prefill 在 warmup 后首次增长 scratch 会真 `cudaMalloc`，只同步当前 GPU 会跨 rank
死锁。收口应在图稳定之后单独做。40–512 KiB 档：图内 NCCL 比 custom one-stage 快
24%+（40 KiB 22.62 vs 29.77；80 KiB 32.15 vs 45.14），而 two-stage 门在 512 KiB，
所以这一档目前仍是 custom，属于已知缺口。

### T7 —— 并发高 C 与长上下文普查（验证缺口，不是新机制）

C=8/16 的固定宽图（方案 §8 的 PR4）**从未验证**，不要把 C=1/C=2 外推；
180K/256K 缺 decode-only 普查（现有 `dec180k.sqlite` 是 prefill 主导，只能当
prefill 证据）；Qwen4-Exp `ForwardBatch` 的 `batch==1` 断言（PR0 主体，散落在
`ForwardTarget`、QSA、GDN、HyperConnection、KV append 与 `graphSequence` 里，
方案估 1–1.5 周）需要 32GB 卡才能端到端验证。

## 5. 各杠杆判定汇总（为什么其余项不再动）

| 项 | 占比（8K / 80K / 180K） | 判定 | 依据 |
|---|---|---|---|
| QPN2 | 55.0% / 46.5% / — | **封顶，不动** | 6.26 ms vs 带宽硬底 5.78 ms，已达 92.4% |
| lm_head GEMV | 6.3% / 5.3% / — | 同带宽墙，**已达 99.8%** | 0.64 GB/卡 ÷ 889 GB/s = 0.72 ms |
| AR | 16.4% / 14.3% / — | 唯一还有量级的杠杆，但**暂缓** | push 中心 +4%–6%，兑现率未知 |
| attention Split | 3.4% / 17.8% / 31.0% | 已定性为 softmax 数学吞吐，**要改算法** | 调常量 −12%，探针见 T4 |
| attention Combine | 2.4% / 2.2% | **已落地** | +10% @ 8K/80K C=1，+4.4% @ 8K C=4 |
| KV 量化 | — | 不作为提速项 | fp4 −4.9%（80K）、−8.2%（180K），越长越差 |
| 并发聚合 | — | 收益取决于 decode 窗口是否重叠 | 8K +0.7% vs 80K +77% |
| norm / swiglu / conv1d | 8.1% / 1.5% / 1.7% | 延迟地板，单 block 且不随上下文变 | 4.7 / 2.7 / 3.7 µs |
| 1Cat pack32 | — | 负收益，禁止 | 10 KiB 24.36 µs vs 现役 17.97 µs，barrier 本身慢 36% |
| XQA / E4M3 | — | 不移植 | 短上下文 attention 5.7%；80K 实测 E4M3 −11.0% |

## 6. 已知风险与运行纪律（都是踩过的）

- **显存峰值比稳态紧得多。** 80K C=1、88 点采样：稳态 14949 MiB、余 1435 MiB；
  **warmup / capture 峰值 16113 MiB、余仅 271 MiB**。最紧的一档是 80K C=1 而不是
  C=2，所以**长 prompt 不能叠 C=4**，也不能靠估算放行（[测]）。
- **QPN2 会在显存压力下静默退回 native**，两条文案在 `fastllm-linear-fp8.cu:3640`
  （`sidecar alloc failed` / `sidecar conversion failed`）。它是正确路径不是错误，
  但会让你的 A/B 悄悄变成"两臂都走 native"。跑前查启动日志、跑后查回退计数。
- **一次观察到的图捕获挂起未定论**：graph on 且四卡 free = 3.23 GB 时卡在
  `warmup capture 2/3 batch=2` 约 8 分钟后被 kill（成功的 run free = 4.36 GB）。
  倾向共租户抢显存，但挂点在 C 单元改过的 `Qwen35MTPLoop` 捕获阶段，
  **判定办法：干净四卡 free ≥ 4.3 GB 重跑**。
- **OOM 的 run 直接作废重跑**，不要当数据点（history：`--tokens` 167936 / 40960
  各 OOM 一次，49152 才过）。
- **跑前确认四卡空闲**（每卡 <1 GB）。有共租户时按时等、不挤；自己的 run 被挤就
  kill 自己的。
- **不要设 `FASTLLM_GPU_TOKEN_HANDOFF=0`**（会换调度器，数字与历史表不可比），
  真 MTP 要用 CLI `--speculative_algorithm mtp --mtp 1`——手工
  `export FASTLLM_QWEN35_ENABLE_MTP=1` 会被 `util.py` 静默覆盖为 0。
- **不要用 `Batch decode after TTFT` 判入队或并发**（§3.5）。
- **不要用 `common window` 单独设门**：它从 last TTFT 起算，按定义惩罚让位——
  80K C=2 的让位代价在 common window 上是 −6.51 tok/s，而 `Total time` 只 +0.7%。
  验收要两个一起看。

## 7. 未验证 / 开放问题（引用数字前先看）

- **严格稳态 decode-only 的 kernel 分布没拿到**：nsys 抓取含 warmup，
  `FASTLLM_SKIP_WARMUP=1` 会把 decode 从 13.31 拖到 29.08 ms/token 且 nsys 报
  `TargetProfilingFailed`。现用两次不同 workload 互相对照补偿。
- **ncu 在本机不可用于算子级归因**：graph=on + 默认 kernel replay 崩（exit 11，
  SIGSEGV 在 `libcuda-injection.so`）；graph=off 死在 save-and-restore（exit 9）；
  唯一能跑的 `--replay-mode application` 拿不到硬件计数器。prefill 的
  `cutlass h884gemm_128x128` 占用 18.8% 与 `volta_h884gemm_256x128` 12.5% 是静态算的，
  **不能据此判定有优化机会**（V100 大 tile GEMM 的常规数字），这一项到此为止。
- **QPN4 是赌注不是结论**：形状门与 QPN2 实际生效门相同（`k%128==0 && n%32==0`），
  不扩覆盖面；融合 epilogue 实测只值 ≤1%；本机无算子级 A/B。
- **Split 的 subgroup=6 假设已被否**，代码已回退、不留开关。
- **并发步长 1.16× vs 2.00× 未裁决**（T2 的 U0）。
- **C=8/16、180K/256K decode-only、`Operand_A_Swizzle_8x64` + ExactMnk tactic
  （PR2-9，对 AWQ 与 NVFP4 M=1 都有用）** 均未做。
- 单次 AR 的 20–69 µs 含排队，无法从 trace 分离（需 CUPTI kernel-level 或 kernel 内计时）；
  `auto-test` 里 NCCL 27.99 µs 比探针 graph 20.11 µs 高 ~8 µs，未闭合（不影响决策）。

## 8. 文档地图

| 文档 | 管什么 | 状态 |
|---|---|---|
| `sm70_1cat_port_plan.md` | 1Cat 移植总排序 + 全部杠杆判定汇总 | 主线，最新实测已回填 |
| `sm70_c2_ttft_overlap_plan.md` | 8K C=2 TTFT 根因链与 U0–U4 | **待执行**（T1/T2） |
| `sm70_tp4_decode_speedup_c2.md` | C=2 方案（M0 口径 / P0 分块 / P1 no-end） | M0 待补、P0 已落地、P1 随 `fa0b6030` 落地 |
| `sm70_tp4_decode_speedup.md` | C=1 方案与「不做」清单 | D1 已落地（`fa0b6030`）、D2 已落地（`08cfe5c4`）、D3 未做 |
| `sm70_tp4_push_ar_plan.md` | TP4 push AR 落地方案 | **暂缓**（人决策），方案保留 |
| `sm70_ar_microbench.md` | AR 微基准（pack32 被否） | 结案 |
| `sm70_ar_microbench_deepdive.md` | AR 成本归因（barrier vs payload） | 结案，no-end 已在引擎兑现 |
| `sm70_npad_ar_chunk_landing_plan.md` | 三单元（C、B、A）落地与验证计划 | 三单元已提交，逐单元 live/perf 门部分待 GPU |
| `sm70_qpn_npad_design.md` | QPN2 侧车 32 列对齐设计 | 已落地 |
| `sm70_nvfp4_coverage_plan.md` | NVFP4 覆盖面收尾与几何 | **待执行**（T3，U1–U4） |
| `sm70_long_prefill_chunk_plan.md` | 长 prompt 串行入队（PR-A/B/C） | PR-A 已落地；PR-B 死路；PR-C 重定性 |
| `sm70_concurrency_port_plan.md` | 并发>1 大方案（PR0–PR5）与实施记录 §16 | PR0/PR1 部分落地，PR2–PR5 未做 |
| `sm70_qwen38nvfp4_1cat_gap_audit.md` | 与 1Cat 的逐项路径对照 | 结案（合同表 + 缺口清单） |
| `sm70_qpn4_qpn8_review.md` | QPN4/QPN8 复看 | 结案（两者都不进默认范围） |
