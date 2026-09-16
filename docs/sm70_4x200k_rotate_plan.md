# 4×200K + FP8 KV + 轮转 优化方案（2026-09-16）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-16GB / TP4 / no-MTP。
前提：4×200K + FP8 KV 在 `--tokens 800000 --max_batch 4 --gpu_mem_ratio 0.98
--low_gpu_mem` 下已能跑完（exit 0，562.18 s，TTFT 138.65/350.30/561.92，
见 `sm70_longctx_200k_2way_fp8.md` §8）。本方案回答一件事：开了
`FASTLLM_PREFILL_ROTATE=1` 之后怎么优化。

## 0. 一句话结论

**轮转买得到 TTFT 公平，买不到吞吐。** 4×200K 的总时长地板是 ~508–562 s，
那是 4 条 200K prefill 在 4 张卡上的算力地板，不是调度器能解决的；轮转只是把
四条请求的完成时间拉齐。想真提速只剩两条路：prefill 内核更快，或每条上下文更短。

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
nsys 自身开销约 21%，干净跑约 83 s / 2166 tok/s）：

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
   NVLink）的结构成本，协议、算法开关都没有可捡的；要降它只有算法级改动
   （reduce-scatter + allgather 的序列并行，砍一半流量，预计可省 prefill
   墙钟 ~15%）或换硬件。
2. **单内核目标是 cutlass NVFP4 GEMM（38%）**，但 80K 时已测过该 GEMM 族
   occupancy 正常（12.5–25%，大 tile V100 GEMM 的正常值，见
   `sm70_1cat_port_plan.md` 5.1），赢它需要真正的 GEMM 工程（tile 形状 /
   split-K 试验），收益不确定。

**追补：Kernel2 摸没摸到算力墙（2026-09-16 实测）——没有，但"cuBLAS 直换"这条
路也被实测否掉了。**

- FLOPs 从 safetensors 形状实算：decoder 14.07B 参数、每卡 3.52B，180K token
  每卡 1.27 PFLOP
- GEMM 内核时间每卡 52.4 s（Kernel2 39.8 s + FP16 GEMM 三族 12.6 s）→ 实际
  **24.2 TFLOPS**；就算把全部 FLOPs 记到 Kernel2 一家头上，上限也只有 31.8
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
- **结论修正**：之前"引擎内核比 cuBLAS 慢 3–4 倍、换 cuBLAS 能拿 −30%"的
  判断是**错归因**。真问题变成：引擎调的就是 cuBLAS、形状也不差，为什么
  在引擎里只有 ~24 TFLOPS 而裸测同样形状有 85–95？当前最强假设是
  **每次调用的分片**：180K 跑里每层每 chunk 有 ~22.5 次 GEMM 发射，明显多于
  该层的投影数，若每次发射只覆盖一小段 token/N，则每次 GEMM 都偏小、效率掉。
  验证工具已留在开关里（打印前 8 个形状与调用计数），下一轮做 FLOPs 统计即可
  裁决。

4a 的落地路线（按性价比排序）：

1. **cuBLAS 直换：已实测否掉**（见上，收益 0）。保留开关只作为诊断/A-B 隔离
   手段，不进默认路径。
2. **fused W4A16 单内核**（省掉 2.1 s 反量化与一次 205 MB 往返）：参考
   1Cat-vLLM 的 AWQ GEMM（`csrc/libtorch_stable/quantization/awq/
   gemm_kernels.cu`，含 `__CUDA_ARCH__ < 750` 分支，sm70 可用）或 cutlass
   W4A8。工程量大于路线 1，作为第二步。
3. **D256 workspace 不适用于 GEMM**：那是 1Cat 的 attention 侧设计（GQA
   head-dim-256 分页注意力 split+combine，`csrc/attention/sm70_v37/
   prefill.cu`），对应的靶子是 attention 桶（12.4%），不是 GEMM 桶（46.7%）。
- 收益上限：GEMM 路径若到墙的一半（~51 TFLOPS），GEMM busy 52.4 → 24.6 s，
  prefill 墙钟约 **−20%**

**Step 4 结论（修正）：** 内核侧有一个有量的目标——NVFP4 GEMM 路径（现在只跑
到实测墙的 1/4 到 1/3），排在序列并行之前、上下文长度之后。此前"没有白捡的
内核优化"的说法作废：occupancy 正常不等于到墙，这一条是被墙实测翻出来的。

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
PCIe 通信决定（chunk 4096 下 4×180K 约 340 s）。要动这个地板只有 §6 收益表里
的三条：Step 4a、Step 4b、砍上下文。任何一步先量再动，A/B 带 sha256 校验。

## 6. 收益总表（最终收益在哪）

场景：4 条 × 180K prompt、200K 池、FP8 KV、chunk 4096。基线：4 条串行 prefill
总计 ≈ 340 s（单条 ~85 s），total 由 prefill 主导。

| 路线 | 最终收益（数字） | 条件与风险 | 状态 |
|---|---|---|---|
| chunk 4096（Step 2） | prefill **−3.8%**，4 条约 **−13 s** | 已实测 sha 同；内存贴边但放得下 | **已采用** |
| Step 1 轮转（现状代码） | **4-way 无收益**：total 559.8 vs 562.2 s 持平，均值 TTFT 350 → 418 s 变差；只有 2-way 买公平 | 2-chunk 准入上限压着 | 4-way 建议关，等 Step 3 |
| Step 3 修 3+ chunk 崩溃 | TTFT 全收敛到 ~total（max 不变），**total 不变**，纯公平 | 崩溃根因未知，代码工作量不确定 | 未做 |
| ~~Step 4a GEMM 换 cuBLAS~~ | **收益 0**（180K 对照 104.54 vs 104.69 s，sha 同、内核发射数逐项相等） | 已实测否掉：prefill 本来就走 dequant + cuBLAS，换枚举无差别。剩下的真问题是"引擎内 24 TFLOPS vs 裸测 85–95"，最强假设是每层 ~22.5 次 GEMM 发射的分片 | 已实现、已否掉 |
| **Step 4b 序列并行**（来自 NCCL 发现） | AR 流量减半 → **prefill −15%，约 −51 s** | reduce-scatter + allgather 大重构，动 norm/激活布局 | 未做 |
| 上下文减半（反向） | **total −50%**，最便宜 | 产品决策 | 随时可拿 |

不叠加说明：4a 与 4b 切的是同一条关键路径，合计现实预期 **−25~35%**，不是
相加的 −35%。

Step 4 明确**不**带来的东西：不改善 TTFT 公平（那是 Step 3/Step 5 的事）、
不省显存（那是 FP4 KV 的事）、不改变 decode。整个方案里真正动 total 的只有
Step 4a、Step 4b 和砍上下文三条；不改代码能拿的只有 chunk 4096（已拿）和
减上下文。
