# 4×200K + FP8 KV + 轮转 优化方案（2026-09-16）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-16GB / TP4 / no-MTP。
前提：4×200K + FP8 KV 在 `--tokens 800000 --max_batch 4 --gpu_mem_ratio 0.98
--low_gpu_mem` 下已能跑完（exit 0，562.18 s，TTFT 138.65/350.30/561.92，
见 `sm70_longctx_200k_2way_fp8.md` §8）。本方案回答一件事：开了
`FASTLLM_PREFILL_ROTATE=1` 之后怎么优化。

## 0. 一句话结论

**轮转买得到 TTFT 公平，买不到吞吐。** 4×200K 的总时长地板是 ~508–562 s，
那是 4 条 200K prefill 在 4 张卡上的算力地板，不是调度器能解决的；轮转只是把
四条请求的完成时间拉齐。

**第七轮收口：内核侧已经没有可捡的杠杆了。** cuBLAS 直换、残差折叠、序列并行、
all-reduce 与计算重叠四条路全部实测为 **0 或负收益**（详见 §6 收益表），根因是
设备侧单流串行（`max_concurrent=1`，四卡各自 107 万内核从不并发），而这段
prefill 由算力 + PCIe 集合通信共同定调。**要动总时长，只剩"减少每条请求的
上下文"这一条**（200K→180K 省 10%、→100K 省 50%，一行配置，无需改代码）。

## 1. 轮转的机制与它买得到什么

`FASTLLM_PREFILL_ROTATE=1` 做两件事（`qwen3_5.cpp:10116-10135`、
`include/models/longPrefillChunk.h`）：

1. 准入预算 `GetBatchedPrefillTokenLimit()` 从"一条 chunk"抬到 **2×chunk**；
2. 排序键换成轮转票，在途请求轮流各喂一条 chunk。

2-way 实测（此前文档，已修复空返回段错误之后）：`inFlight=2` 出现 97 轮，
TTFT 253.3 / 254.6 s，**Total time 与串行相同（约 255 s）**。这是铁证：
重叠两条 prefill 不降总时长，因为单条 200K prefill（~127–139 s）已经把 4 张卡
占满，两条只是轮流用同一份算力。

所以 4-way 的实测（Step 1）：成对公平，不是全收敛；总时长与串行持平
（559.77 vs 562.18 s）。

## 2. 硬约束：一次 forward 最多 2 条 chunk，是实测崩溃

代码注释（`qwen3_5.cpp:10127-10129`）原话："three or more chunks in one forward
abort in the batched prefill path (measured)"。即 3+ chunk 同 forward 会在批量
prefill 路径里崩，所以"4 条同时进一个 forward"不是配置能开的事，要改代码。
2 条是每天被 batch-2 路径反复走过、安全的上限。

这条约束直接封死了"把 4 条塞进一个 forward"这条路线；能不能开 3+/4× 取决于那
个崩溃的根因是不是一处可以修的调度/状态账（待定位，见 Step 3）。

## 3. 分步方案（每步带验证门）

### Step 1. 轮转基线实测（2026-09-16，已完成）

同 4×200K 配置 + `FASTLLM_PREFILL_ROTATE=1`：**exit 0，wall 595 s，无 OOM、
无 illegal address、无段错误**。此前文档里"开着自己会崩"的说法在当前构建已不
成立（`56e3445f` 修复生效）。对照同配置的串行跑：

| 指标 | 串行 | 轮转 |
|---|---:|---:|
| Total time | 562.18 s | **559.77 s**（基本持平）|
| TTFT min / avg / max | 138.65 / 350.30 / 561.92 s | **276.60 / 418.47 / 559.44 s** |

TTFT 的形状是"两条一组"：约 [277, 277, 554, 554] s，不是四条收敛到同一时刻。
原因就是 §2 的 2-chunk 上限，同一时刻只有两条请求在途：先到的一对各 ~2×单条
完成，后到的一对排到 ~4×。所以 4-way 下轮转买的是**成对公平**，同时把第一条
TTFT 从 139 s 拖到 277 s、平均值变差（350 → 418 s）。要不要这个交换是产品
语义（Step 5），不是性能。

### Step 2. prefill chunk 尺寸 A/B（2026-09-16，已完成）

测试场景：200K 上下文池（`--tokens 200000`）+ 80K prompt（`--input_tokens 80000`）、
单条、FP8 KV、`--low_gpu_mem`、ratio 0.98。chunk 四档：

| chunk | Prefill | TTFT | Total time | 校准后 4 卡 gpuFree | 相对 2048 多用/卡 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2085 tok/s | 38.37 s | 38.48 s | 20362 MB | — |
| 4096 | 2166 tok/s | 36.94 s | 37.06 s | 19354 MB | 252 MB |
| 8192 | **2233 tok/s** | **35.82 s** | **35.95 s** | 16986 MB | 844 MB |
| 16384 | 2229 tok/s | 35.89 s | 36.02 s | 12330 MB | 2008 MB |

四档 sha256 全同（`1d23759c…`），数值透明。读数：

1. **chunk 越大 prefill 越快，膝盖在 8192**：2048→8192 吞吐 +7.1%（2085 →
   2233 tok/s）、TTFT −6.6%（38.37 → 35.82 s）；16384 与 8192 持平（噪声内）。
   调度轮从 39 降到 10，总时长只省 7%，说明瓶颈是 GEMM 算力不是调度开销，
   "少轮次就能明显变快"的预期不成立。
2. **内存代价随 chunk 加速**：每翻倍约多 252 / 844 / 2008 MB/卡。4×200K 场景
   校准后每卡只剩 326 MB，只有 2048 稳、4096 贴边、8192 放不下；要用 8192 得
   先砍池子（`--tokens` 降档）腾出 ~0.85 GB/卡。
3. **结论**：这个杠杆的天花板约 7% prefill 时长（4×200K 下约 35 s），值得在
   内存允许时拿，但不是量级改变。4×200K 的下一步是把池子降一点换 chunk 4096/
   8192，或者接受 2048。

### Step 3.（代码）定位并修复"3+ chunk 崩溃"

把一次 forward 的准入上限从 2 抬到 3/4 条 chunk 的唯一前提。

- 步骤：复现（`FASTLLM_SCHED_TRACE` + 抬高预算的探针）→ 定位崩溃点
  （先在批量 prefill 的 recurrent-state/空返回路径查，56e3445f 已修过一处）→
  修复 → 加回归测试（照 7b726cb4 的做法）。
- 验证门：4×200K + 4×chunk 预算 exit 0；同时裁决一个悬案——如果总时长仍
  ≈ 4×单条，说明 GPU 已饱和、重叠无益，这个改动就只买公平；如果下降，说明
  之前 2-way 的重叠没真正并行，这里还有吞吐可挖。
- 风险：如果崩溃点牵涉批量 prefill 的生命周期/状态账（GPMMU 那类），工作量
  可能不小，先做最小复现再排期。

### Step 4. prefill 内核提速（真正的吞吐杠杆）

总时长地板的唯一来源是 prefill 算力。本节先修正两点，再动手：

1. **测量工具是 nsys，不是 ncu。** 本机记录（.audit）已证实 ncu 的 kernel
   replay 会撞 FastLLM 的 graph 捕获期记账（哪怕只过滤 prefill 内核，也会在
   warmup 崩，exit 6/11）；application replay 完成但拿不到计数器。nsys
   2026.1.3 是这台机器全程在用的工具，80K 的 prefill 归因（cutlass h884 GEMM
   44.6%、volta FP16 GEMM 30.1%、cuBLAS paged attention 11.8%、NVFP4 反量化
   6.7%，NCCL 另计 37%）就是从 `dec80k_post.sqlite` 用 sqlite 查询拿到的。
2. **80K 的占比不能搬到 180K。** 16 层全注意力对增长前缀做 attention，代价
   O(context²)；GEMM 只随 context 线性长，所以 180K 下 attention 的相对份额
   会翻倍以上。必须按真实配置（180K prompt、chunk 4096）重新采样。另外原稿
   写的 AR push 是 decode 侧杠杆，对 prefill 无效，移除。

实测（2026-09-16，`/home/nsys/pf180k.sqlite`，180K prompt + chunk 4096 +
200K 池 + FP8 + low_gpu_mem；nsys 下 Total 105.37 s / Prefill 1711 tok/s，
nsys 开销极小（同配置干净跑实测 104.54 s / 1725 tok/s，与轨迹的 105.37 s
只差 ~1%）：

每卡 kernel busy 108.4 s / 118.2 s 窗口（91.8%，多流重叠）。归因（每卡）：

| 内核族 | 每卡时长 | 占 busy | 说明 |
|---|---:|---:|---|
| **cutlass Kernel2（NVFP4 GEMM）** | **39.8 s** | **36.7%** | 253696 次发射（4 卡合计），平均 628 us |
| NCCL AllReduce（RING_LL） | 35.8 s | 33.0% | 每卡 6656 次、平均 5.38 ms、消息 41.9 MB |
| attention（causal softmax + update + gather） | 13.4 s | 12.4% | 80K 时占 compute 11.8%，平方增长符合预期 |
| FP16 GEMM（s884/h884 三族） | 12.6 s | 11.6% | |
| NVFP4 反量化 | 2.1 s | 1.9% | 80K 时 6.7%，占比下降 |
| 其余 | ~4.7 s | ~4% | |

两个判定：

1. **NCCL 不是肉。** 41.9 MB 消息、5.38 ms 一次 → busbw ≈ 11.7 GB/s，已是
   PCIe3 x16 实用峰值（~12–13 GB/s）的九成。这 33% 是 TP4 over PCIe（无
   NVLink）的结构成本，协议、算法开关都没有可捡的。
   （本节写的时候以为"reduce-scatter + allgather 能砍一半流量、省 ~15%"，
   **第七轮实测已否掉**：RS+AG 每 rank 环上流量与 all-reduce 相同、还慢 6%，
   见 §6。）
2. **单内核目标是 cutlass NVFP4 GEMM（38%）**，但 80K 时已测过该 GEMM 族
   occupancy 正常（12.5–25%，大 tile V100 GEMM 的正常值，见
   `sm70_1cat_port_plan.md` 5.1），赢它需要真正的 GEMM 工程（tile 形状 /
   split-K 试验），收益不确定。

**追补：Kernel2 摸没摸到算力墙（2026-09-16 实测）——没有，但"cuBLAS 直换"这条
路也被实测否掉了。**

- FLOPs 从 safetensors 形状实算（**第三次更正后的正确数**）：decoder GEMM 参数
  **24.35B**（embed 1.271 + lm_head 1.271 + mtp 0.425 + decoder 24.35 ≈ 27.3B，
  与 27B 命名一致），每卡每 token 12.18 GFLOP，180K token 每卡 **2.19 PFLOP**。
  此前写过的 14.07B / 1.27 PFLOP / "与引擎发射量差 1.88 倍"**全是错的**：
  一是 NVFP4 权重按**打包形状**数了参数（`weight_packed` 一个字节装 2 个 fp4，
  如 `in_proj_qkv.weight_packed [10240, 2560] U8` 实为 10240×5120），每条打包
  投影都少数一半；二是层结构并非均匀（实际 **48 层 GDN + 16 层全注意力**）。
  修正后引擎发射 2.390 PFLOP 对需求 2.192 PFLOP = **仅 9% 余量，没有冗余算力**
- GEMM 内核时间每卡 52.4 s（Kernel2 39.8 s + FP16 GEMM 三族 12.6 s）→ 按引擎
  计数器的真实发射量算 **45.6 TFLOPS**（Kernel2 单独 61.0）
- 同卡实测可达墙（`/tmp/gemmbench.cu`、`/tmp/gemmbench3.cu` 可复跑）：cuBLAS
  fp16 tensor op 在 N=512–32768 是 78.5–104 TFLOPS；换成引擎真实形状
  （cublas M=k、N=2048、K=5120 与 K=1536）仍是 **85–95 TFLOPS**，且
  `CUDA_R_16F` 与 `CUBLAS_COMPUTE_32F` 两种累加精度几乎一样（差 3% 以内）
- **cuBLAS 直换实测（已实现、已验证、收益为 0）**：加了
  `FASTLLM_NVFP4_PREFILL_CUBLAS` 开关（默认关），它让 prefill 绕过
  TurboMind/Marlin 并把算法枚举换成 TENSOR_OP。180K prompt 对照：
  Total **104.54 s（关）vs 104.69 s（开）**，token sha256 相同
  （`adcb1bd7…`），nsys 两条轨迹的内核发射数**逐项相等**
  （Kernel2 253696、反量化 49152、s884gemm 177408）
- 原因（代码 + 轨迹双重确认）：**prefill 本来就走在 dequant + cuBLAS 这条
  路上**。TurboMind 的 `GemmNvfp4` 在 n>16 时不参与（开关打开后计数器显示
  我这版路径调用数千次、`chunks=1`；关掉时 TurboMind 计数器一次不打），而
  轨迹里那个 `cutlass::Kernel2<cutlass_70_tensorop_h884gemm_128x128_tn_align8>`
  是 cuBLAS 内部的内核（TurboMind 派发表里没有这些名字）。且这版 cuBLAS 上
  `CUBLAS_GEMM_DEFAULT` 与 `_TENSOR_OP` 选中同一批内核，所以换枚举无差别
- **结论修正（第二轮实测，2026-09-16）**：上面"分片"假设与"24 TFLOPS"两个
  说法都被推翻，实测把真问题定到了别处：

  1. **没有分片**。形状直方图（开关打开时 atexit 打印）显示每次调用就是一次
     完整投影：8K 跑 6144 次 / 5 种形状、180K 跑 49152 次 / 10 种形状，
     `flops/call` = 60–365 GFLOP，180K 的 49152 次 = 44 chunk × 64 层 × 4.36，
     正好是该层的投影数。
  2. **FLOPs 必须按形状算**。180K 跑 GEMM 路径每卡 **2.39 PFLOP**
     （4 卡合计 9564 TFLOP），除以 GEMM 内核时间 52.4 s = **45.6 TFLOPS**
     （Kernel2 单独 61.0）。而按权重形状（含打包 fp4 展开）算出的**需求**是
     2.192 PFLOP/卡，即引擎只多跑 9%——**没有冗余算力可挖**。此前用参数量
     （14.07B）估出的需求 1.27 PFLOP、"差 1.88 倍"两个说法都作废（打包权重被
     少数一半 + 层结构按均匀估错，见上一条）。
  3. **不是功耗墙**。同形状裸测连续 60 秒稳定 **97.3 TFLOPS**，1432 MHz、
     243 W（上限 300 W），无降频。
  4. **真问题是通信与计算零重叠**。device 0 上 63424 个 GEMM 内核与 NCCL
     重叠的是 **0 个**，attention 与 NCCL 重叠也是 0 个；39.2 s GEMM +
     36.3 s NCCL + 13.3 s attention 严格串行相加，整跑 util 91.8%、
     空隙只有 9.7 s。也就是说 GPU 没有在计算时把 36 s 的 all-reduce 藏起来。

  **所以 4a 的真靶子从"换内核"改成了"把 all-reduce 藏进计算"**：这不是
  GEMM 内核问题，是 TP4 在 PCIe 上的调度/重叠问题。

4a 的落地路线（按性价比排序）：

1. **cuBLAS 直换：已实测否掉**（收益 0）。保留开关只作为诊断/A-B 隔离手段，
   不进默认路径。
2. **all-reduce 与 GEMM 重叠：第七轮实测否掉**（详见 §6）。当时的估计是把
   36.3 s 的 NCCL 藏进 39.2 s 的 GEMM 背后、上限 104 → ~68 s（−35%）；
   第六轮记的"分块重叠上限 14–15%（≈5–6 s）"**来自合成探针，不是本引擎实测**（R1 未实现，见 §8）：该数只说明"探针里两条流能叠"，不能代表引擎。引擎实测到的是队列深度 1
   （`max_concurrent=1`），只换流拿不到任何收益，要动就得重排 TP 行并行的
   算子级流水线——5 s 的收益配不上这个改动面。
3. **fused W4A16 单内核**（省掉 2.1 s 反量化与一次 205 MB 往返）：参考
   1Cat-vLLM 的 AWQ GEMM（`csrc/libtorch_stable/quantization/awq/
   gemm_kernels.cu`，含 `__CUDA_ARCH__ < 750` 分支，sm70 可用）或 cutlass
   W4A8。收益量级只有几秒，且按第七轮的根因（工作量被重新分配而非消除），
   预期同样中性——未做。
4. **D256 workspace 不适用于 GEMM**：那是 1Cat 的 attention 侧设计（GQA
   head-dim-256 分页注意力 split+combine，`csrc/attention/sm70_v37/
   prefill.cu`），对应的靶子是 attention 桶（12.4%），不是 GEMM 桶（46.7%）。

**Step 4 结论（第七轮，最终）：** GEMM 内核本身没有 3 倍空间（45.6–61 TFLOPS
对实测 97.3 墙，且需求侧只差 9%）；NCCL 也不是"没藏好"那么简单——引擎运行在
**队列深度 1**（四卡各自 107 万内核同时最多 1 个在飞、重叠 0.00 s；主机 worker
线程 86% 的时间被同步挡住）。异步派发开关对主机同步**没有可测影响**（更正见
§8.0：文档此前写的"降 92%（388.1 → 32.5 s）"是两个不同负载相除的结果，同负载
下开关前后同步次数逐项相等）——所以"主机阻塞不是障碍"这个结论仍然成立，但支撑
它的那条证据作废。真因已定位到 SM70 的 per-AR 主机同步，见 §8。分块重叠（每层输出切块、AR 走独立流、两侧有可交换的独立
工作）在理想提交模型下的上限是 **14–15%**（nsplit=8 最优，已串正确依赖），
落到 104.2 s 上的收益**没有实测**。唯一有引擎支撑的上界是**稳态段**的设备空闲时间约 **3.1 s**（§8.3；早先记的 8.6 s 把 warmup 段算进去了，已更正），R1 只会比它小。要动就得重排 TP 行并行的算子级流水线。
4b 的"RS+AG 流量减半"经实测为错（环上流量相同，还慢 6%）。
此前所有版本估算（"没有白捡的内核优化"、"GEMM 只跑到墙的 1/4"、
"−35%/−23%/−6%"）全部作废。

**追补：残差折叠（第七轮，已实现、实测中性、收为 opt-in）。**

新增 `FastllmCudaHalfMatMulFloatNVFP4Block16AddTo` 与 `DoCudaLinearAdd` 的 NVFP4
分支（开关 `FASTLLM_NVFP4_LINEARADD`，**默认关**），让 prefill GEMM 用
`cublas beta=1` 直接累加进残差，省掉独立的 `middle` + `AddTo`：

- **数值安全**：引擎真实形状（cublas M=8704/N=4096/K=5120、`CUDA_R_16F` 累加、
  TENSOR_OP）下 `beta=0 + __hadd` 与 `beta=1` **逐位相同**（3570 万元素 0 差异）；
  四臂实测 token sha256 全同（`adcb1bd7`）。
- **机制生效**：内核普查 `FastllmAddToKernel` 8576 次 / 1.070 s →
  2432 次 / 0.004 s；计数器确认省掉 755 GB 残差访存。
- **但墙钟中性**：四臂交替（off/on/off/on）Total 103.9547 / 103.9466 /
  104.3836 / 104.2355 s，off 均值 104.1692 vs on 均值 104.0910，
  差 **+0.078 s（+0.075%）**，而运行间波动 0.437 s——**效应仅为噪声的 0.18 倍**。
- **根因（工作量重新分配，非消除）**：同期 NCCL all-reduce 从 36.10 s 涨到
  37.05 s（+0.95 s），Kernel2/volta GEMM 各 +0.05 s，**设备总忙碌恒定
  （108.45 vs 108.47 s）**。省下的 1.07 s 基本原封不动出现在集合通信里。
  AR 单次 +2.6%（5.423 → 5.566 ms）不是额外交付（两者都已在 10.8–11.1 GB/s
  有效带宽，PCIe 上限 ~12.5），属 peer-wait。
- 注意：本机 `max_concurrent=1`（单设备单流），所以不能说"AddTo 与通信并发"，
  只能说"工作量被重新分配"。
- **延伸预测**：该折叠"连反量化一起融"的变体同理无收益——这段路由集合通信
  定调，被移除的访存不是瓶颈。

### Step 5. TTFT 策略决策（轮转开还是关）

轮转的代价是惩罚早到请求。2-way 实测 #0 的 TTFT 从 127.4 s 变慢到 253.3 s。
所以：

- 四条长请求**几乎同时到达**：开轮转，公平性好，总时长不变。
- **错开到达**或对"第一条尽快出"有要求：关轮转，串行让 #0 最快。
- 这个决策不属于性能，属于产品语义，需要按真实到达模式定。

## 4. 反向思考：目标本身要不要 4×200K

- "4 条并发、每条最多 200K"和"4 条全都 200K"是两回事。平均上下文降到 100K，
  总时长直接减半，这是比任何内核优化都便宜十倍的选择。
- FP4 KV 只省内存（池子 6.56 → 3.7 GB/卡），不降 prefill 算力；且本机实测
  fp4 在越长上下文越亏（180K −8%）且 fp4 tiled 路径每档都更慢，不优先。
- 2×TP2 真并行：200K 在 TP2 下内存放不下（权重×2、KV×2），否决。

## 5. 预期结论

4×200K 已实测稳定（exit 0），TTFT 可选公平或最快，total 地板由 prefill 算力 +
PCIe 集合通信共同决定（chunk 4096 下单条 180K 实测 104.5 s，4 条约 418 s）。
**内核侧的四条路（cuBLAS 直换、残差折叠、序列并行、all-reduce 重叠）已全部
实测为 0 或负收益**，原因一致：设备侧单流串行（`max_concurrent=1`），这段
prefill 由算力 + PCIe 通信定调，局部优化改不动墙钟。§6 收益表里唯一剩下的
真实杠杆是**上下文长度**。任何一步先量再动，A/B 带 sha256 校验。

## 6. 收益总表（最终收益在哪）

场景：4 条 × 180K prompt、200K 池、FP8 KV、chunk 4096。基线：4 条串行 prefill
总计 ≈ 418 s（单条实测 104.5 s，4 条串行），total 由 prefill 主导。

| 路线 | 最终收益（数字） | 条件与风险 | 状态 |
|---|---|---|---|
| chunk 4096（Step 2） | prefill **−3.8%**，4 条约 **−16 s** | 已实测 sha 同；内存贴边但放得下 | **已采用** |
| Step 1 轮转（现状代码） | **4-way 无收益**：total 559.8 vs 562.2 s 持平，均值 TTFT 350 → 418 s 变差；只有 2-way 买公平 | 2-chunk 准入上限压着 | 4-way 建议关，等 Step 3 |
| Step 3 修 3+ chunk 崩溃 | TTFT 全收敛到 ~total（max 不变），**total 不变**，纯公平 | 崩溃根因未知，代码工作量不确定 | 未做 |
| ~~Step 4a GEMM 换 cuBLAS~~ | **收益 0**（180K 对照 104.54 vs 104.69 s，sha 同、内核发射数逐项相等） | 已实测否掉：prefill 本来就走 dequant + cuBLAS，换枚举无差别 | 已实现、已否掉 |
| **Step 4a′ all-reduce 藏进计算**（第六轮，已否掉） | **不动了**：主机阻塞已证明不是障碍——异步派发开关开启后主机同步**无可测变化**（更正见 §8.0；原文"降 92%"是 180K/8K 两个负载相除，同负载实测开关前后 `cudaStreamSynchronize` 逐项相等），**四卡并发达不到 2、墙钟不变**。设备侧根本不给并发机会：所有工作排在同一条流上。要在 4×180K 上拿到分块重叠的收益（未测；稳态上界约 3.1 s，§8.3），得重排算子级流水线（把每层行并行拆块、NCCL 走独立流、两侧有可交换的独立工作），工作量与收益（收益未测，稳态上界约 3.1 s）严重不成比例 | 需要动 TP 行并行的算子级流水线；收益未实测（稳态上界约 3.1 s，即 104 s 的约 3%），而风险（数值、死锁）真实存在 | **否掉，不再做**；但真因（SM70 的 per-AR 主机同步）已定位，见 §8 |
| ~~Step 4b 序列并行~~ | **前提错误，作废**：实测 RS+AG 与 all-reduce 流量**相同**（环上各 1.5S/rank），41.94 MB 下 5.874 ms vs 5.567 ms（慢 6%） | 真要减流量得按 S/n 分片激活（序列并行的完整形态），是另一种更大的重构，与"RS+AG 替换 AR"不是一回事 | 否掉 |
| ~~残差折叠（`FASTLLM_NVFP4_LINEARADD`）~~ | **收益 0**：四臂交替 off 104.1692 vs on 104.0910 s，差 +0.078 s，仅噪声（0.437 s）的 0.18 倍；AddTo 8576→2432 次但同期 NCCL +0.95 s，设备总忙碌恒定 | 已实现、数值逐位安全（sha 全同），但工作量只是被重新分配；连同其"连反量化一起融"的延伸变体一并否掉 | 已实现、默认关、否掉 |
| 上下文减半（反向） | **total −50%（约 −209 s）**，最便宜 | 产品决策 | 随时可拿 |

不叠加说明：4a′（分块重叠）与 4b（序列并行）切的是同一条关键路径，不能相加；
但按实测两者都已被否掉，这条说明只对"若将来重启其中一条"有意义。

**结论：收益表里唯一还剩的真实杠杆是"上下文长度"。** 所有内核侧路线
（cuBLAS 直换、残差折叠、序列并行、all-reduce 重叠）都已实测为 0 或负，
根因一致：这段 prefill 由 prefill 算力 + PCIe 集合通信共同定调，设备侧单流
串行（`max_concurrent=1`），任何局部内核/访存优化都不会改变墙钟。

Step 4 明确**不**带来的东西：不改善 TTFT 公平（那是 Step 3/Step 5 的事）、
不省显存（那是 FP4 KV 的事）、不改变 decode。**第七轮收口后的实情是：内核侧
四条路全否掉，真正能动物理总时长的只剩"减少上下文"这一条**；不改代码能拿的
只有 chunk 4096（已拿）和减上下文。

## 7. 本轮新增的开关、工具与构建环境

### 两个实验开关（都默认关，不动默认路径）

| 开关 | 作用 | 状态 |
|---|---|---|
| `FASTLLM_NVFP4_PREFILL_CUBLAS=1` | 让 prefill 的 NVFP4 GEMM 绕过 TurboMind/Marlin、算法枚举换 `_TENSOR_OP`；并打印按形状统计的调用/FLOPs 直方图（atexit） | 收益 0，仅作诊断与 A-B 隔离 |
| `FASTLLM_NVFP4_LINEARADD=1` | 让 prefill GEMM 用 `beta=1` 直接累加进残差，省掉独立 `AddTo`；带调用/流量计数器 | 收益 0，默认关；保留作为"连反量化一起融"的基座 |

### 可复跑的测量工具

- `/tmp/pf_attr.py`：nsys sqlite 的内核归因（按时长排序、拆 eager/graph、分出 NCCL）。
- `/tmp/gemmbench.cu` / `gemmbench3.cu`：cuBLAS fp16 形状扫描，用于量本机可达墙。
- `/tmp/gemm_sustain4.cu`：四卡并发持续 GEMM，用于排除功耗/散热天花板。
- `/tmp/overlap_split.cu`：分块重叠（nsplit 扫描），已串正确依赖。
- `/tmp/overlap_dep.cu`：依赖链下四档 A/B/C/D。
- `/tmp/fused_norm_bench.cu`：融合 add+norm 对两趟链的同形状微基准。
- `/tmp/rsag2.cu`：all-reduce vs reduce-scatter+all-gather 的流量与时延对比。
- `/tmp/hide2.cu`：GEMM/AR 互相隐藏的上界与依赖链形态（§8.3 主表）。
- `/tmp/hide3.cu`：并发时 GEMM 与 AR 各自变慢多少（争用归因）。
- `/tmp/decomp.cu`：跨卡共享 vs 单卡内部争用的分解（四卡同跑 +0.0%）。
- `/tmp/tflops.cu`：固定窗口吞吐计数（单跑 143.7 → 并发 111.7 TFLOPS/卡）。
- `/tmp/longconc.cu` + `/tmp/samp.sh`：长跑并发并采样功耗/时钟（排除功耗墙）。

### 构建环境（本轮装好，后续迭代受益）

装了 **ccache 4.5.1** 并重配构建：

```sh
cmake -S . -B build-sm70-tests -DUSE_CUDA=ON -DCUDA_ARCH=70 -DUNIT_TEST=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

受控实测（删掉 `.o` 后重编 `fastllm_tools`）：**关缓存 44 s、开缓存 2 s**；
真实改一行代码 45 s（两种 miss），撤回该行 1 s（`direct_cache_hit` + 
`primary_storage_hit`）。缓存 `/root/.ccache`，上限 30 GB，配置在 `ccache.conf`。

三条使用要点：`-j` 用 **10** 合适（`nproc`=10）；单个 nvcc 峰值内存仅
0.2–0.6 GB（共 30 GB），线程数不受内存限制；**新声明尽量留在 `.cu` 内**——
碰 `fastllm-cuda.cuh` 这类公共头会扇出 25 个编译单元（约 20 倍代价）。

## 8. 复核：异步派发开关的"降 92%"是错的；并发=1 的真因（2026-09-16 追补）

本节写**三件有据可查的事**：纠正 §3/§6 里那句错的引文；把并发=1 的真因定位到
代码行；以及 **R1（分块 + AR 独立流）的引擎实测结果（§8.7）**。

> 状态更新（2026-09-16 晚）：R1 **已经实现并跑通**（§8.7）。本节早先写的
> "R1 没有实现，不给收益数字"已作废，保留在这里作为撤错留痕。

### 8.0 更正："降 92%" 不成立（引擎实测）

| 读数 | 值 | 来源 |
|---|---:|---|
| 文档里的 388.1 s | 180K prompt，`/home/nsys/pf180k.sqlite`，428 万内核 | 开关**缺席**的那份 |
| 文档里的 32.5 s | 8K prompt，`/tmp/asy8k.sqlite`，54.8 万内核 | 开关 **ON** 的那份 |

**两者是不同负载，不是开关的前后对照。** 内核数相差 7.81 倍，同步量自然相差 12 倍。

同负载实测（引擎 trace）：

| 场景 | `cudaStreamSynchronize` 次数 | 总阻塞 |
|---|---:|---:|
| 8K，`ASYNC_DISPATCH=0` | 6920 | 32.41 s |
| 8K，`ASYNC_DISPATCH=1` | 6920 | 32.45 s |
| 180K，flag 缺席 | 60848 | 388.14 s |
| 180K，`ASYNC_DISPATCH=1` | 60848 | 387.57 s |

**同负载下开关前后逐项相等**（差 0.04 s / 0.6 s，都在噪声内）。

另一处更正：文档说"`MultiCudaSetPersistentAsyncDispatch` 仅 deepseekv4.cpp 启用、
qwen3_5.cpp 从未启用"——**不准确**。`src/models/qwen3_5.cpp:426` 有
`Qwen35ScopedMultiCudaAsyncDispatch` 类，`:27439` 在 MLP 路径实例化它，且该类
**早于**加开关的提交 `c2fb222d`。

**仍然成立**：四卡并发是 1（180K trace 四卡各自 `max_concurrent=1`、
`>=2 内核在飞的时间 = 0.0000 s`）；主机 worker 线程 86% 时间被同步挡住。
"主机阻塞不是障碍、设备侧串行才是"这个**方向对**，但支撑它的证据作废。

### 8.1 并发=1 的真因（代码 + 引擎实测）

1. **SM70 上每次集合通信后强制主机同步。** `fastllm-multicuda.cu:2450` 的
   `FastllmNcclPostSyncEnabled()` 在 `FastllmCudaGetNcclForceSync()` 为真时返回真，
   于是 `ncclAllReduce` 发射后立刻 `cudaStreamSynchronize`（`:3210`，另有三个集合
   通信同样处理）。而 `basellm.cpp:4093` 的 warmup 收尾**按架构设门**：

   ```cpp
   if (FastllmCudaRuntimeArch() >= 75) { FastllmCudaSetNcclForceSync(false); }
   ```

   SM70（arch=70）**不满足**，所以 `ncclForceSync` 全程 `true`，每次 AR 都同步。
   原注释写明原因：SM70 长 prefill 会在 warmup 后首次增长 chunked-attention /
   dequant scratch，真实 `cudaMalloc` 只同步当前 GPU、排不空别的 rank 的在途
   NCCL，会跨卡死锁。

2. **GEMM 与 AR 在同一条流上。** 80K 引擎 trace device 0：stream 558 同时装着
   FP16_GEMM 254496 次 + NCCL_AR 3456 次 + Kernel2 15592 次；两条主流的
   **交替次数只有 1**（stream 46 在 12.05 s 结束，stream 558 从 12.41 s 才开始），
   即"先后两段"，不是"并行两路"。

**抬掉第 1 条的门（实验开关 `FASTLLM_SM70_NCCL_ASYNC=1`，见 §8.4），引擎实测：**

| 场景 | 同步次数 | 总阻塞 | `max_concurrent` | 墙钟 | sha256 |
|---|---:|---:|---:|---:|---|
| 80K，门保持 | 23120 | 152.87 s | 1（四卡） | 38.1410 s | `1d23759c` |
| 80K，`SM70_NCCL_ASYNC=1` | **12800（−45%）** | **82.06 s（−46%）** | **仍 1** | 38.2326 s | `1d23759c` |

**同步腰斩，但并发仍 1、墙钟不动。** 第 1 条是必要不充分：次数降下来了，
GEMM 和 AR 还在同一条流上排队。**降主机阻塞 ≠ 买并发。**

### 8.2 收益表（R1 已实现并实测，收益为**负**；R5 已判定**不该做**）

单位统一为 80K prompt（`--input_tokens 80000 --output_tokens 8`），四张 V100、TP4、
`FASTLLM_QWEN35_SM70_CUDA_GRAPH=0`，同一台机器同一时段串行跑。五次运行 sha256 全部
为 `1d23759c`：

| 配置 | Total | Prefill | 相对对照 |
|---|---:|---:|---:|
| 对照（不开 R1） | 37.3322 s | 2151.76 tok/s | — |
| R2：AR 放独立流、不分块 | 39.9610 s | 2009.59 tok/s | **+7.04%** |
| R2 + 还原 host drain（`FASTLLM_TP_AR_SIDE_STREAM_HOST_SYNC=1`） | 38.2152 s | 2101.80 tok/s | **+2.37%** |
| R1：AR 独立流 + `nsplit=4` 分块流水 | 39.3311 s | 2041.97 tok/s | **+5.35%** |
| R1 + 还原 host drain | 39.4692 s | 2034.73 tok/s | **+5.72%** |

**R2 那 7 个点拆开了**（这也是本节前一版写错的地方，见下面的撤错）：
还原每个集合通信后的 host drain，就把 **7.04 点里的 4.37 点**拿回来
（39.9610 → 38.2152）。剩下 **2.37 点**才是"换一条流"本身的代价
（每次集合通信两次 `cudaEventRecord` + 两次 `cudaStreamWaitEvent`，
外加主流要等 `side.done`）。
而 R1 的分块重叠只买回 **1.58 点**（39.9610 → 39.3311），**小于这 2.37 点的地板**，
所以怎么叠都翻不了正。

| # | 路线 | 实测收益 | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|---|
| R1 | AR 移出主流 + 分块流水 | **−5.35%（80K）**；其中分块重叠本身**是正的**：相对 R2 回收 **1.58 点** | 已实现 | 数值次序变化（本模型 hash 未变）；rank 门控不同步会**死锁** | **已实现、已实测、默认关、不划算** |
| R2 | 只把 AR 放独立流、不分块 | **−7.04%**（80K）。其中 **4.37 点**来自去掉 host drain，**2.37 点**来自换流的固定开销 | 已实现 | — | 已实现、已实测、默认关 |
| **R5** | **让自定义 one-stage all-reduce 支持侧流，再叠 R1 的分块** | **已实现并实测，结论是不要用**：8K 上比同一配置（`splits=8`）**慢 6.5%**；80K 上直接 **`cudaErrorIllegalAddress` 崩掉**（exit 134） | 需要 `FASTLLM_TP_AR_CUSTOM_AR_ON_SIDE_STREAM=1` + `FASTLLM_CUDA_CUSTOM_ALLREDUCE=1` 同时打开，且分块必须 ≤ 8 MiB | 非默认流上的二阶段自定义内核非法访存；机制未坐实 | **已实现、已实测、默认关、崩溃，禁止开启** |
| R3 | 抬掉 SM70 force-sync 门（`FASTLLM_SM70_NCCL_ASYNC=1`） | 同步 **−45%**，**并发仍 1、墙钟 0**（引擎实测，§8.1） | 已实现 | 跨 rank 死锁（SM70 理由仍在） | 已实现、仅实验、**默认关** |

**撤错留痕（本表的前两版）**

第一版（更早）：把 R2 那 7 个点归因于"侧流路径必须关掉自定义 one-stage
all-reduce，改用 NCCL"，并据此把 R5 列为"唯一还值得做的方向"。**归因是错的**，
来自只读了 `fastllm-multicuda.cu` 里那句 `allowCustomAllReduce=false` 就外推，
没有去查自定义 AR 在本机到底有没有被启用。

第二版（本轮前半）：改口说 R5 的"收益恒为 0，因为分块张量 10 MiB 超过
`CustomArMaxBytes() = 8 MiB`，连强制模式都进不去"。**这一版也错了**：10 MiB 是
`splits=4` 的结果，`splits=8` 时每片只有 5 MiB，在上限之内。本轮把 R5 真正实现
并测了，实测结果见上面 R5 行与 §8.8。教训是同一条：**先把"目标状态可达"验证掉，
再谈收益**，而且不要用一次算例代替全部算例。

**结论**：**R1 能藏住延迟（1.58 点），但侧流这条路的地板是 2.37 点，藏不回来。**
要压掉那 2.37 点只能去掉事件同步，而事件同步正是流间定序的唯一手段，
去掉就是 §8.7 那个死锁。R5 走的是"把自定义 AR 搬上侧流"这条绕行路，
实测既慢又崩（§8.8）。**所以 R1/R2/R5 这条线结案：全部默认关，不再投入。**

### 8.3 唯一有引擎支撑的上界：设备空闲时间

只读引擎自己的 180K trace，不依赖任何未实现的东西：

**注意：这里必须只看稳态段。** 全部 118.2 s 窗口里含约 12.4 s 的 warmup
（权重加载 / 预热），那一段空闲多、但不属于那条 180K prefill：

```
全段    : busy 108.45 / span 118.20 = 91.8% 忙 -> 空闲 9.70 s（含 warmup）
稳态段  : 只取 >=12.4 s 的窗口
          busy 102.10 / span 105.27 = 97.0% 忙 -> 空闲 3.17 s
       -> 折算到干净跑 104.2 s：约 3.1 s，即总量的约 3%
```

**含义**：稳态段设备已 **97.0%** 忙，所以**即使把全部等待填满，空间也只有约 3.1 s**
（早先本行写的 8.6 s 是把 warmup 的 6.6 s 一起算进来的错误口径，已更正）。
这是"并发最多能带来多少"的**上界**，不是 R1 的预期收益；R1 只会比它小。

**而且这 3.17 s 里大半填不了。** 稳态段 >10 ms 的间隙共 44 个，其中 **43 个**的
前后内核组合是同一个：`ncclDevKernel_AllReduce_Sum_f16_RING_LL` →
`ncclDevKernel_Broadcast_RING_LL`（每次约 34.5 ms）。这是 **NCCL 内部**
allreduce→broadcast 的交接，不是"GPU 在等主机"，挪算子流填不进去。

### 8.4 本轮新增的开关

| 开关 | 作用 | 默认 | 状态 |
|---|---|---|---|
| `FASTLLM_SM70_NCCL_ASYNC=1` | 覆盖 `basellm.cpp` 的 SM70 架构门，让 warmup 后 `ncclForceSync=false`。仅用于并发实验 | **关** | 已实现、引擎实测同步 −45%、并发不变 |
| `FASTLLM_TP_AR_SIDE_STREAM=1` | 把 TP all-reduce 丢到每 (卡,线程) 一条侧流上，主机不再为每次集合通信阻塞 | **关** | 已实现、已实测（§8.2 的 R2） |
| `FASTLLM_TP_AR_SIDE_STREAM_PIPELINE=1` | 在 R2 基础上把行并行输出按行分块，第 c 片的 AR 盖住第 c+1 片的 GEMM | **关** | 已实现、已实测（§8.2 的 R1） |
| `FASTLLM_TP_AR_SIDE_STREAM_SPLITS=N` | 分几块，默认 4 | 4 | — |
| `FASTLLM_TP_AR_SIDE_STREAM_HOST_SYNC=1` | 把侧流路径在发射后本来刻意省掉的 `cudaStreamSynchronize` 加回来。**只用于拆 R2 那 7 个点**，不是能上线的模式 | **关** | 已实现、已实测（§8.2，把 7.04 点拆成 4.37+2.37） |
| `FASTLLM_TP_AR_CUSTOM_AR_ON_SIDE_STREAM=1` | R5：允许自定义 one-stage all-reduce 跑在侧流上。需同时 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=1` 且分块 ≤ 8 MiB 才生效 | **关** | 已实现、已实测：8K 慢 13.2%，**80K 非法访存崩溃**，禁止开启（§8.8） |
| `FASTLLM_CUSTOM_AR_CENSUS=1` | 自定义 AR 的接单普查：按四条理由（超尺寸上限 / 策略拒绝 / 指针登记拒绝 / 真正发射）计数并统计重放字节，退出时打印 | **关** | 已实现。**判断"自定义 AR 有没有参与"的唯一可靠手段**（§8.8） |
| `FASTLLM_TP_AR_DEBUG=1` | 每次 Begin/End 与每个分块打一行到 stderr。**卡死时唯一的定位手段** | **关** | 已实现 |

### 8.5 已经做完的三步（原计划）

1. **在引擎里实现 R1 的最小形态**——已完成（§8.7）。
2. **量 `t_>=2` 是否 >0**——**未做**。本轮改用了一个更直接、更便宜的证据：
   死锁与修复都由探针计数判定（`[R1dbg-p]` 四卡是否对称），见 §8.7。
   nsys 的 `max_concurrent` 计数仍待补，但 §8.2 已给出墙钟结论，不影响取舍。
3. 量墙钟——已完成：80K 上 R1 比对照慢 5.35%（§8.2）。

### 8.6 复跑方式

```sh
# 8K 匹配对照（§8.0）
PYTHONPATH=build-sm70-tests/tools FASTLLM_MULTICUDA_ASYNC_DISPATCH=0 \
  python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --dtype auto --tokens 200000 --max_batch 1 --gpu_mem_ratio 0.98 \
  --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size 4096 \
  --input_tokens 8192 --output_tokens 8 --batch 1 --warmup 0

# 并发实验（§8.1）：同样的命令，前面加
FASTLLM_SM70_NCCL_ASYNC=1
```

并发用 sqlite 扫：`max_concurrent` = 对某 device 的内核区间做扫描线取峰值；
`t_>=2` = 在飞内核数 ≥2 的时间总和。

### 8.7 R1 实现记录：一个"门控不同步"导致的四卡死锁（2026-09-16）

这一节是 §8.2 那三行数字背后的过程。写它的原因：**这个 bug 的表现是"GPU1 卡死"，
而进程既不报错也不退出**，第一次碰只能靠逐卡利用率看出来。

#### 8.7.1 现象

跑 `FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 FASTLLM_TP_AR_SIDE_STREAM=1
FASTLLM_TP_AR_SIDE_STREAM_PIPELINE=1` 时进程不退出、不打错误，日志停在
`TP AR chunked pipeline: engaged with 4 splits (1024 rows/chunk)`，逐卡利用率是

```
GPU0 100%   GPU1 0%   GPU2 100%   GPU3 100%
```

TP 集合通信挂住时的指纹就是这条：**卡住的那个 rank 没有任何内核（0%），
另外几个在 NCCL 里空转（100%）**。而且卡住的是哪张卡每次不一样
（先 GPU1，后 GPU3），所以不是"某张卡上的逻辑分支写错了"，是竞态/次序不一致。

对照：同样的命令**不开 R1** 时 44 s 正常跑完、hash 对。所以问题在 R1。
再二分：**只开 `FASTLLM_TP_AR_SIDE_STREAM=1`、不开 PIPELINE** 也是正常的
（1536 次集合通信走侧流，3.5108 s）。所以问题在**分块那一段**。

#### 8.7.2 定位手段：两行探针 + 一个看门狗

用 `timeout` 干等是浪费（第一次白等了十几分钟）。改成：

* `tools/gpu_watchdog.sh` 包住命令，连续 4 次采样出现"≥2 卡在跑、≥1 卡 util=0"
  就判定卡死，**立刻**抓现场（逐卡利用率、目标进程每个线程的 `wchan`/`syscall`、
  日志尾部、Xid 前后计数）并杀掉。45 s 内就拿到了结论。
* `FASTLLM_TP_AR_DEBUG=1` 让每次 `Begin`/`End` 和每个分块各打一行探针，
  **带 device id**。卡死时每个 rank 的最后一行就说明它停在哪一步。

探针给出的关键读数：

| 读数 | dev0 | dev1 | dev2 | dev3 |
|---|---:|---:|---:|---:|
| 分块流水线探针行数 `[R1dbg-p]` | **234** | **0** | **0** | **0** |
| 侧流集合通信次数（`Begin`） | 106 | 30 | 36 | 36 |

**分块流水线只跑在 0 号卡上，1/2/3 号卡一次都没进去。**
0 号卡每层发 4 次集合通信（分 4 片），另外三张每层发 1 次（不分块）。
NCCL 是**按位置配对**的，第 2 个位置就把"1024 行的分片 reduce"和"4096 行的整体
reduce"配到了一起，于是挂住。

#### 8.7.3 根因：那道门读了 rank 独有的状态

`Qwen3CudaTryTpChunkedLinearResidualReducePipeline` 里原来的门控有一项：

```cpp
middle.dims.back() != weight.dims[0]      // 旧代码
```

`middle` 就是 `buf.mlpPart`。它**只在 rank0 的 fallback 分支里被创建**：
rank0 走 `Qwen3CudaLinearAddBlock`（内部会 `Qwen3CudaPrepareLocalOutput(middle)`），
而 1/2/3 号卡的 fallback 是 `Qwen3CudaLinear(runner, input, weight, bias, hiddenStates)`
——**直接把 GEMM 写进残差，从不碰 `middle`**。所以 1/2/3 号卡的 `middle.dims`
**整轮都是空的**，门控永远拒绝它们。

形状打印（`FASTLLM_TP_AR_DEBUG=1` 时在拒因处各卡打前 3 次）证实了这一点：
四张卡报的形状**完全一致**（`w=[5120,1536]`、`res=[1,1,5120]`、`in=[1,1,1536]`），
只有 `mid=[-]`（空）。

**教训**：原代码的注释写着"这个判据是 rank 不变的"，理由是
"`rows, splits, chunk, dtype` 每张卡都一样"——这句是对的，但那个判据**还读了
`middle.dims`**，而它是 rank 独有的。**跨 rank 的判据只能读跨 rank 一致的状态**，
这一点必须在代码里写死，不能靠"我检查过了"。

#### 8.7.4 修法

1. 门控只读**跨 rank 一致**的量：`hiddenStates`（残差，各卡相同）、`input` 的
   数据类型与行数、`weight.dims.size()`、环境变量。**删掉对 `middle.dims` 的依赖。**
2. `middle` 改由流水线**自己定形**（`PrepareLocalOutput` + `Resize` + `Allocate`，
   照 `PrepareLocalOutput` 在别处的既有用法），**在 fork 任何集合通信之前**完成
   ——这样"某张卡要分配内存"不会打乱集合通信的次序。
3. 探针里补上 device id：拒因、形状、每个分块都带 `dev=`。原来的计数是
   进程级共享的，一个 rank 静默回退和四张卡都回退看起来一模一样。

修完的对称性证据（8K，同样开关）：

| 读数 | dev0 | dev1 | dev2 | dev3 |
|---|---:|---:|---:|---:|
| `[R1dbg-p]` 行数 | 3456 | 3456 | 3456 | 3456 |
| 分块调用次数 | 384 | 384 | 384 | 384 |

`P:join` 共 1536 = 384 × 4，四卡完全对称；hash 与对照一致，看门狗判定 OK。

#### 8.7.5 结论

* R1 **能跑通、数值正确**（8K 与 80K 的 sha256 均与对照一致）。
* R1 的**分块重叠本身是有效的**：相对"只开侧流"回收 **1.58 点**。
* 侧流路径的净代价是 **2.37 点**（每次集合通信的事件定序 + 主流等 `side.done`）。
  叠加后 R1 净结果 **−5.35%**。**R1 单独不划算，默认关。**
* 本节前一版把侧流的代价记成 7 点、并归因于"丢掉了自定义 all-reduce"。
  **已更正**：7 点里 4.37 点是去掉 host drain，2.37 点才是换流的开销。
  自定义 all-reduce 在本机**默认**不参与 prefill（见 §8.8 的实测）。

### 8.8 R5 实现记录：自定义 one-stage all-reduce 走侧流（2026-09-16 深夜）

**做了什么。** 给自定义 AR 加了一条流参数，再让侧流路径可以用它：

* `FastllmCudaCustomAllReduce(data, dest, count, dataType, deviceId, stream)`——
  `stream == nullptr` 就是原来的行为（调用线程的默认流）；传自己的流就排到那条流上。
  捕获期间拒绝非默认流：图捕获路径把发射录在默认流上，混用会破坏图。
* `LaunchCustomAr<T>` 与 `RunCustomArCandidate` 逐层透传该流，**包括 in-place
  scratch 的 copy-back**（那一处如果漏了，结果会被拷到另一条流上）。
* 开关 `FASTLLM_TP_AR_CUSTOM_AR_ON_SIDE_STREAM=1`（默认关）。

**真正的接线障碍不是"内核写死了自己的流"，而是 `allowCustomAllReduce`。**
`FastllmNcclAllReduceOnStream` 一直传 `allowCustomAllReduce=false`，所以不论
`FASTLLM_CUDA_CUSTOM_ALLREDUCE` 怎么设，侧流路径连自定义 AR 的入口都进不去。
这一点是 census 抓出来的（见下），不是读代码读出来的。

**新增了一个可复用的量具：census**（`FASTLLM_CUSTOM_AR_CENSUS=1`，默认关）。
自定义 AR 有四个静默拒绝的理由，只看时间分不清"跑在自定义内核上"和"悄悄交给
NCCL"。census 按理由计数并统计重放字节数，退出时打印。它本轮直接纠正了我两次判断：

```
# 接错线时（8192 次全来自默认流的非重叠集合通信）
    declined: over size cap / type          0 call(s),       0.0 MiB
    launched on the custom kernel        8192 call(s),     210.0 MiB
# 接对线后（侧流上的分块也进去了）
    launched on the custom kernel       20480 call(s),   61650.0 MiB
```

**实测 1：80K 上强制自定义 AR、但**不走**侧流（即 R1 关）**

| 读数 | 值 |
|---|---|
| Total / sha256 | **37.0933 s** / `1d23759c`（对照走 NCCL 是 37.3322 s） |
| census：超尺寸上限被拒 | **10752 次，420480 MiB** |
| census：跑在自定义内核上 | 8192 次，210 MiB（平均 26 KiB） |

两件事被**测出来**了：① 引擎默认路径上，**所有 40 MiB 的 prefill 张量都因为
`CustomArMaxBytes() = 8 MiB` 被拒**，一字节都没走自定义内核（10752 × 40 MiB =
420480 MiB 全部退回 NCCL）；② 自定义 AR 在它够得着的那些小张量上并不亏
（37.0933 vs 37.3322，略快且 hash 一致）。

**实测 2：R5（侧流 + 自定义 AR），8K，全部 sha256 `adcb1bd7`**

| 配置 | Total | 相对对照 |
|---|---:|---:|
| 对照（不开 R1） | 3.2568 s | — |
| R1 `splits=8`（不开自定义 AR） | 3.4882 s | +7.11% |
| R5，接线错误（自定义 AR 只在默认流上生效） | 3.4688 s | +6.51% |
| **R5，接线正确（自定义 AR 真上了侧流）** | **3.9495 s** | **+21.3%** |

所以 R5 相对同配置（`splits=8`）**慢 13.2%**。

**实测 3：R5 在 80K 上崩，而且**把三张卡打到设备级异常**。**
`R1 + splits=8 + 自定义 AR 走侧流`，80K：`exit=134`，日志给出
`CUDA error = 700, cudaErrorIllegalAddress`（`fastllm-multicuda.cu:194`），
四张卡各报一次，随后 `terminate`。更严重的是内核日志：

```
[11039.499526] NVRM: Xid (PCI:0000:05:00): 13, Graphics SM Warp Exception on
               (GPC 5, TPC 4, SM 0): Out Of Range Address
[11039.501310] NVRM: Xid (PCI:0000:08:00): 13, Graphics Exception: ESR ...
[11039.503102] NVRM: Xid (PCI:0000:07:00): 13, Graphics Exception: ESR ...
[11039.523185] NVRM: Xid (PCI:0000:07:00): 43, pid=916085, name=python3, Ch 00000008
[11039.545157] NVRM: Xid (PCI:0000:05:00): 43, pid=916085, name=python3, Ch 00000008
```

`tools/gpu_watchdog.sh` 的跑前/跑后计数把归属钉死了：这次 **Xid 3 → 330**（新增
327 行），而**同一次会话里之后两次运行都是 330 → 330**：

| 运行 | 配置 | Xid 跑前→跑后 | 结果 |
|---|---|---|---|
| R5 80K | R1 `splits=8` + 自定义 AR 走侧流 | **3 → 330** | exit 134 |
| 判别 80K | 强制自定义 AR，**不走侧流**（R1 关） | 330 → 330 | exit 0，37.0933 s，hash 对 |
| 回归 8K | 全默认 | 330 → 330 | exit 0，3.2544 s，hash `adcb1bd7` |

所以这个 SM 越界**只出现在自定义 AR 跑在非默认流上时**，不是自定义 AR 本身的问题，
也不在默认路径上。事后四张卡健康：Retired Pages 0、单双比特 ECC 0、
Pending Page Blacklist No、温度 47–49 °C、空闲 0 MiB，且随后两次完整四卡运行
输出哈希都正确。

机制**没有坐实**，只记边界：出问题时 `splits=8` 使每片 5 MiB，`useTwoStage` 为真，
因此这是引擎里**第一次**把二阶段自定义内核放在非默认流上、以 in-place 方式跑
prefill 尺寸的张量。是否与二阶段内核的跨 rank 屏障、或与
`BuildCustomArRegistration` 里那次阻塞 `cudaMemcpy` 的流归属有关，未验证。

**安全性**：该路径要同时打开 `FASTLLM_TP_AR_CUSTOM_AR_ON_SIDE_STREAM=1` 和
`FASTLLM_CUDA_CUSTOM_ALLREDUCE=1` 才会走到，默认全部关闭，默认路径不受影响。
**但因为它会让三张卡报设备级 SM 越界，代码里已按"禁止开启"写死在注释里**，
保留开关只为让这个失败可复现、可修。

**回归证据**（`build-sm70-tests/customAllReduceRegression`，四种配置全 PASS）：

| ranks | `FASTLLM_CUDA_CUSTOM_ALLREDUCE` | 结果 |
|---:|---|---|
| 2 | 1（强制） | PASS，enabled=1，selected/tested_paths=6 |
| 2 | 0（关闭） | PASS，enabled=0 |
| 4 | auto | PASS，enabled=0（策略在 TP4 上关掉它） |
| 4 | 1（强制） | PASS，enabled=1，selected/tested_paths=6 |

说明这次透传流参数的改造对默认流上的行为是保持的。


## 9. 补测 §8.5 缺的那一项：R1 下的 `t≥2`（2026-09-17）

§8.5 第 2 条自己记着"量 `t_>=2` 是否 >0——**未做**"。本节把它补上，结果**推翻了
§8.1/§8.3 的前提**。

### 9.1 实测

配置：`FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 FASTLLM_TP_AR_SIDE_STREAM=1
FASTLLM_TP_AR_SIDE_STREAM_PIPELINE=1 FASTLLM_TP_AR_SIDE_STREAM_SPLITS=8`，
80K prompt、TP4、4×V100、chunk 4096、FP8 KV，device 0。

| 轨迹 | `max_concurrent` | `t≥2` | 占窗口 |
|---|---:|---:|---:|
| **R1（侧流 + 分块 8）** | **2** | **10.5546 s** | **23.31%**（窗口 45.27 s）|
| 对照：180K 无 R1（`pf180k.sqlite`）| 1 | 0.0000 s | 0% |

并发度的时间分布（R1）：`c=0` 5.664 s / `c=1` 29.052 s / **`c=2` 10.555 s**，
三者之和 = 窗口 45.27 s；`∫并发度 dt = 50.161 s = Σ各内核时长`（互查通过）。

四条流：`419` 29.78 s（GEMM/attention 侧）、**`423` 17.77 s（集合通信侧）**、
`46` 1.64 s、`412` 0.90 s。stream 423 上那 20480 个内核是
`grid=2x1 block=288`、由 `cuLaunchKernelEx` 发起，即 NCCL all-reduce 的形状。

工具自证：同一个扫描线脚本先在两条已知答案的轨迹上跑过
（`pf180k` → `1 / 0.0000 s`，`dec80k` → 与文档记录一致）才用于本节。

### 9.2 含义：前提被推翻，结论不变

- **被推翻**：§8.1/§8.3 写的"队列深度 1、换流拿不到任何收益""要重叠必须先改主机
  提交模型"。实测是：**换流之后设备上真的出现了两个内核同时飞，占窗口 23.3%**。
  "并发不可能发生"这个前提是错的。
- **不变**：R1 仍然是净亏。同场 A/B 实测：对照 **37.6385 s** vs R1 **40.4062 s**
  （**+7.4%**）。所有臂 sha256 都是 `1d23759c81d9f3e7`。
- 所以正确的说法是"**并发发生了，但没有变成墙钟收益**"，而不是"并发不可能"。

### 9.3 同场待补（中断）

同场 A/B 本要跑四臂（对照 / R2 / R1-s4 / R1-s8），对照完成后进程中途被一起
GPU 硬件事故打断（见 §9.4），只拿到对照一个数。**R2、R1-s4、R1-s8 三臂未完成、
无数据。**

### 9.4 事故：GPU 掉卡（红线级，已停手报告）

跑 R2 那一臂时四张卡全部掉线。内核日志：

```
NVRM: Xid (PCI:0000:07:00): 79, ... GPU has fallen off the bus.
NVRM: Xid (PCI:0000:07:00): 154, GPU recovery action changed to 0x1 (GPU Reset Required)
NVRM: Xid (PCI:0000:08:00/0000:09:00): 154 ... GPU Reset Required
pcieport 0000:00:02.0: AER: Multiple Uncorrectable (Non-Fatal) error message received
nvidia 0000:05:00.0/07:00.0/08:00.0/09:00.0: AER: can't recover (no error_detected callback)
NVRM: Attempting to remove device 0000:05:00.0 with non-zero usage count!
```

`nvidia-smi` 报 `Unable to determine the device handle for GPU0..3: Unknown Error`、
`No devices were found`；受害进程抛 `terminate called after throwing an instance of
char const*`。

**时间线**：崩溃日志早于我的作业取消；此前对照臂已正常完成。**本机同时有另一个
代理在做 FlashInfer `pcie_ipc`，从本会话内部无法归因触发者。**

**处置**：按 `fastllm/AGENTS.md` 的红线，**未对卡做任何动作**——没有 `kill -9`
持卡进程、没有 `sleep` 后重试、没有 `nvidia-smi -r`、没有 PCI remove/rescan，
只做了一次 `nvidia-smi`/`dmesg` 读取取证后停手报告。**需要人工介入。**
