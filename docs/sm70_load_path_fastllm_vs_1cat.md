# 模型加载模块对照：FastLLM (SM70/TP4/QUASAR-NVFP4) vs 1Cat-vLLM 1.5.0

只读代码分析。**未运行任何 benchmark / 测试 / 编译。**

- 目标场景：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4
- 参考基准：`/tmp/1Cat-vLLM-1.5.0`（vLLM 1.5.0 的 V100/SM70 分支）
- 对照实现：`/home/fastllm`（baseline `7d5ebea5`）

**证据等级**（本文只允许三种）：**【实测】**（目标系统上跑出来、有产物）、**【引自文档/日志】**（别人或历史 run 记录的数）、**【我的推断】**（我读代码 + 推算，没跑）。本次分析没跑任何东西，因此绝大多数是后两种。

---

## A. 两份实现的加载流程要点

### A1. FastLLM 侧（每条带 file:line）

1. **入口**：`CreateLLMModelFromHF` 是 HF safetensors 目录的加载入口（`src/model.cpp:4438`）。
   只解析 `model.safetensors.index.json` 拿到分片文件名列表（`src/model.cpp:4496-4507`），**不 mmap 数据文件**。
2. **只读 safetensors header**：`SafeTensors` 构造函数对每个分片 `fopen` + 读 8 字节长度 + 读 JSON header，只为建 name → (file, offset, shape, dtype) 的字典（`src/model.cpp:1735-1754`）；`GetSortedItemNames()` 按 (文件名, offset) 排序（`src/model.cpp:1756-1768`）。
3. **逐 tensor 读盘**：`SafeTensorItem::CreateBuffer` 用 `fopen`+`fseek`+`fread` 把每个张量读进**新分配的堆缓冲**（`src/model.cpp:1614-1703`）。**没有 mmap，没有 posix_fadvise 预读**（全仓 grep 只在 disk-MoE 设备上有一处 fadvise：`src/devices/disk/diskdevice.cpp:602`）。
4. **并行读**：线程数 `threadNum = min(16, max(4, GetAlivePool()->threads.size()))`（`src/model.cpp:4890`），按累计字节切分工作区间（`src/model.cpp:4939-4953`），非流式张量用 `threadNum` 个 std::thread 并行（`src/model.cpp:5517-5523`）。
5. **NVFP4 block-16 的加载期转换**：读 packed 字节 + 读 `weight_scale`（E4M3），然后**逐 block 把 FP8 scale 展开成 inline FP32** 写进新缓冲（`src/model.cpp:1246-1263`，`float curScale = fp8e4m3tofp32.dict[scaleByte] * scale2Value;`）。存储从 9 B/16elem 变 12 B/16elem（`src/fastllm.cpp:761-763`）。
6. **落到 CPU Data**：`CreateFromOriData`（`src/model.cpp:5263-5271` → `src/fastllm.cpp:1461-1502`）；对未被修改的浮点张量有 `TryAdoptSafeTensorBuffer` 直接接管 tensor 的 buffer，省一次全尺寸分配（`src/model.cpp:2133-2166`）。
7. **按层流式（已落地）**：`ShouldLoadWeightSeriallyBeforeOthers` 把每层张量挑成串行组（`src/models/qwen3_5.cpp:25352-25366`），`GetWeightLoadPriority` 返回 `100000 + layer`（`src/models/qwen3_5.cpp:25325-25335`）。开关判定要求 dense + 全 CUDA 设备映射 + 单卡时非 low-mem/非 CPU-KV（`src/models/qwen3_5.cpp:25134-25173`）。
8. **组内并行、组间串行**：串行组按优先级分批，批内起 `min(threadNum, 组大小)` 个线程（`src/model.cpp:5480-5512`）；每批结束回调 `OnWeightLoadGroupFinished()`。
9. **组内完成即 TP 切分 + H2D 并释放 CPU 源**：`PrepareStreamingTpLayer` / `PrepareStreamingSingleCudaLayer`（`src/models/qwen3_5.cpp:24906` / `:24866`）；组结束再 `malloc_trim(0)` 把加载期临时分配还给 OS（`src/models/qwen3_5.cpp:25544-25550`）。
10. **TP 切分细节**：`SplitMultiCudaWeight` 先把源搬到 root device，再从 host 或 root device 按 division scheme 逐片 `cudaMemcpy` 到各卡（`src/devices/multicuda/fastllm-multicuda.cu:1027`、`:1056-1066`、`:1136-1350`），NVFP4 的行/列切分有专门分支（`:1278-1316`、`:1317-1340`）。
11. **QPN2 侧车在 warmup 后一次性建**：`OnAutoWarmupFinished` 遍历所有 `NVFP4_BLOCK_16` 权重调 `FastllmCudaWarmupNvfp4Qpn2Sm70`（`src/models/qwen3_5.cpp:9685-9720`）；真正打包在 `FastllmCudaEnsureNVFP4Qpn2Layout`（`src/devices/cuda/linear/fastllm-linear-fp8.cu:3627-3659`），`nvfp4Qpn2Wanted` 是粘性标记，防止 TurboMind 就地重排破坏 native 布局（同文件 `:3595`）。
12. **native .flm 格式**才走 `WeightMap::LoadFromFile`，那里才有 `USE_MMAP` 分支（`src/fastllm.cpp:3445-3450`，`FileMmap` 实现 `:823-837`）。**编译开关默认关**（`CMakeLists.txt:17`）且当前 SM70 构建确认是 `USE_MMAP:BOOL=OFF`（`build-sm70-tests/CMakeCache.txt:259`）。

### A2. 1Cat 侧

1. **统一入口**：`BaseModelLoader.load_model` 三步走 —— `initialize_model` → `load_weights` → `process_weights_after_loading`（`vllm/model_executor/model_loader/base_loader.py:43-82`）。
2. **加载器可插拔**：16 种 `load_format`（`load_format` Literal + 分派表，`vllm/model_executor/model_loader/__init__.py:34-68`），含 safetensors 变体、gguf、bitsandbytes、tensorizer、modelexpress、runai_streamer、sharded_state。
3. **默认 safetensors 路径**：`safe_open(st_file, framework="pt")` 逐张量 `f.get_tensor(name)`（`weight_utils.py:1072-1079`）。`safe_open` 是 mmap 读取，`get_tensor` 拷出到新 torch tensor —— 所以它**也不是零拷贝**，但省掉了一次显式文件读。
4. **可选多线程整文件读**：`multi_thread_safetensors_weights_iterator` 用 `ThreadPoolExecutor(max_workers=4)` + `load_file(device="cpu")`（`weight_utils.py:1082-1109`），由 `enable_multithread_load` / `num_threads` 控制（`default_loader.py:299-311`）。
5. **可选 page-cache 预读**：`_prefetch_all_checkpoints` 起后台线程按块读一遍文件；网络 FS（NFS/Lustre）且 checkpoint 能装进 RAM 时自动开（`weight_utils.py:831-920`、`:961-1005`）。
6. **逐层完成即处理**：`initialize_online_processing` 给每层包一层 `online_process_loader`，用 `load_numel >= load_numel_total` 判断"这层读完了"（`reload/layerwise.py:120-205`），到齐后 `_layerwise_process`：上设备 → 装载 → 量化后处理 → 拷回原 tensor 存储（`reload/layerwise.py:333-370`）。注意这条路径服务于**在线量化/权重 reload**，不是普通的 NVFP4 checkpoint 加载路径。
7. **NVFP4 重排在加载末尾一次做掉**：`process_weights_after_loading` 里 TurboMind 打包（`sm70_turbomind.py:289-345`）和 QPN2 打包（`compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:373-449`），做完把源参数替换成空张量释放（同文件 `:352-357`、`:442-449`）。
8. **只保留一份打包布局**：QPN2 侧车只存 codes + **原始 E4M3 scale 字节**（`nvfp4_qpn2_sm70.cu:366-403`），FP8→FP16 在 kernel 里做（同文件 `:105-110`）；大 M 走 QPN2-packed prefill 分派（`nvfp4_qpn4_sm70.cu:663-678`，阈值 `VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M = 1024`，`vllm/envs.py:1777`）。
9. **实测加载数字（日志产物）**：`Loading weights took 13.25 s` + draft `2.71 s`，`Model loading took 11.08 GiB and 20.477 s`，`init engine (profile, create kv cache, warmup model) took 62.03 s (compilation: 46.76 s)`（`/tmp/serve_e4m3_native.log:98,128,131,238`）。

---

## B. 差异清单表

| 维度 | FastLLM 现状 | 1Cat 做法 | 差距是否要紧 |
|---|---|---|---|
| **NVFP4 是原样搬还是加载即变换** | 加载期把 E4M3 scale 展开成 inline FP32 block-16（`src/model.cpp:1246-1263`）→ 12 B/16elem | 保留原始 packed + raw E4M3 scale（`nvfp4_qpn2_sm70.cu:105-110` 在 kernel 里反量化）→ 9 B/16elem | **要紧，但只是显存**。见 B-1 的字节账。不影响正确性 |
| **反量化发生在哪个阶段** | 权重：加载期一次（scale 展开）；激活：每次前向（M>32 prefill 走 dequant+cuBLAS，`src/devices/cuda/linear/fastllm-linear-fp8.cu:3902-3970`） | 权重：kernel 内；prefill 也有 packed 路径（`nvfp4_qpn4_sm70.cu:663`） | 权重侧不是差距；prefill 算力是另一条线（`sm70_4x200k_rotate_plan.md:166-175` 已结案：prefill 本来就走 dequant+cuBLAS） |
| **mmap / 直读 safetensors** | HF 路径 **没有** mmap（`src/model.cpp:1614-1703` 是 fopen/fread）；只有 native `.flm` 有 mmap，且开关默认关（`CMakeLists.txt:17`、`CMakeCache.txt:259`） | `safe_open` 走 mmap（`weight_utils.py:1072`），可选后台预读进 page cache（`weight_utils.py:848-918`） | **不要紧**。见 C-3 |
| **是否分块流式加载** | **是**（layer-wise，已落地）：按优先级分组、组内并行、组间串行、组尾 `malloc_trim`（`src/model.cpp:5480-5512`、`src/models/qwen3_5.cpp:25544-25550`） | 有一个**面向 reload/在线量化**的逐层 meta 机制（`reload/layerwise.py`），普通加载路径并不按层释放；它靠"重排后把源置空"降峰值 | **FastLLM 在这一点上更强**（这是已拿的收益，见 C-0） |
| **加载与 TP 切分的顺序** | 读一个张量 → 立刻切分并 H2D → 释放 CPU 源；切分时先落 root device 再分发（`fastllm-multicuda.cu:1056-1066`） | 每个 rank 独立读自己需要的分片，`tensor_parallel_size=4` 时 `weight_loader` 直接按 axis 切片装到本 rank 的 CUDA 参数里 | 顺序合理，无差距；但**跨卡读盘量不同**，见 B-2 |
| **加载期显存峰值：是否先全量落一份再切** | **不是**。逐张量/逐层切完就搬，CPU 源在组尾释放 | 每个 rank 只物化自己的分片 | 无差距 |
| **加载期 GPU 侧常驻：几份 NVFP4 布局** | **两份**：native block-16（cudaData）+ QPN2 侧车（nvfp4Qpn2Packed）。设计注释明说"不要就地覆盖 cudaData"（`fastllm-linear-fp8.cu:3563-3565`） | **一份**：打包完把源参数置空（`compressed_tensors_w4a4_nvfp4.py:352-357`、`:442-449`），大 M 由 packed-prefill 分派兜住 | **要紧**，见 C-1 |
| **是否支持"加载即重排"** | **支持**，且只做一次：warmup 后一次性建全部侧车（`src/models/qwen3_5.cpp:9685-9720`），`nvfp4Qpn2Wanted` / `IsRepacked` 是粘性门（`fastllm-linear-fp8.cu:1445-1485`、`:3595`） | **支持**，且只做一次：`process_weights_after_loading`（`base_loader.py:80`） | **没有差距。前向期不存在重复 repack。** 差异只在"何时做"和"是否释放源" |
| **host 侧 staging 缓冲** | **有**一个明确的额外拷贝：开 `USE_MMAP` 时，H2D 之前先 `new uint8_t[expansionBytes]; memcpy(...)` 把 mmap 页拷进堆（`src/fastllm.cpp:2768-2774`）。当前构建该分支不编译 | 没有显式 staging；`get_tensor` 直接从 mmap 拷进 torch tensor | 不要紧（当前构建根本没走这条路） |
| **加载耗时主要花在哪** | 逐张量 `fread`（19.147 GiB 全量）+ E4M3→FP32 scale 展开 + TP 切分 cudaMemcpy H2D。没有预读，读和 H2D 不重叠 | 逐张量 `get_tensor`（mmap 拷贝）+ 加载后一次性重排（TurboMind/QPN2 prepack kernel）| 见 C-2 / C-3 |

### B-1. NVFP4 字节账（【我的推断】，算法写在下面）

按真实 checkpoint header + FastLLM 的 TP4 切分轴算。切分轴已对代码核实：`axis=0`（切输出 N）用于 `q/k/v_proj`、`in_proj_qkv`、`in_proj_z`、`gate/up_proj`；`axis=1`（切输入 K）用于 `o_proj`、`out_proj`、`down_proj`（`src/models/qwen3_5.cpp:24972-24973` 的 `splitLinear(oWeightName, …, oScheme, 1, …)`；down 同族）。

```
packed codes / rank     = Σ N_local × K_local / 2
raw E4M3 scales / rank  = Σ N_local × K_local / 16          (1 B / 16elem)
QPN2 sidecar / rank     = Σ ceil32(N_local) × K_local × (1/2 + 1/16)
native block-16 / rank  = Σ N_local × (K_local/16) × 12      (inline FP32 scale)
```

| 投影（每 rank） | 张数 | N_local | K_local | native GB | 侧车 GB |
|---|---:|---:|---:|---:|---:|
| `mlp.gate_proj` | 64 | 4352 | 5120 | 1.070 | 0.802 |
| `mlp.up_proj` | 64 | 4352 | 5120 | 1.070 | 0.802 |
| `mlp.down_proj` | 64 | 5120 | 4352 | 1.070 | 0.802 |
| `linear_attn.in_proj_qkv` | 48 | 2560 | 5120 | 0.472 | 0.354 |
| `linear_attn.in_proj_z` | 48 | 1536 | 5120 | 0.283 | 0.212 |
| `linear_attn.out_proj` | 48 | 5120 | 1536 | 0.283 | 0.212 |
| `self_attn.q_proj` | 16 | 3072 | 5120 | 0.189 | 0.142 |
| `self_attn.o_proj` | 16 | 5120 | 1536 | 0.094 | 0.071 |
| `self_attn.k_proj` / `v_proj` | 32 | 256 | 5120 | 0.032 | 0.024 |
| `linear_attn.in_proj_a` / `b` | 96 | 12 | 5120 | 0.004 | 0.008 |
| **合计** | **496** | | | **4.566 GB**（4.252 GiB） | **3.430 GB**（3.194 GiB） |
| **两份同时常驻** | | | | | **7.996 GB**（7.446 GiB = 16 GiB 卡的 46.5%） |

**独立交叉校验（不依赖切分假设）**：checkpoint header 里 U8 = 11.339 GiB、F8_E4M3 = 1.417 GiB，合计 **12.757 GiB = 13.697 GB**。而这 496 个张量的 packed + scales 在全模型口径下 = `Σ N×K/2 + Σ N×K/16` = 13.697 GB ✓。逐 rank 值 ×4 = 13.70 GB，两边对上。

（张数口径：本表按**未合并**的 checkpoint 张量计 496 条（gate/up 分开）；仓库文档按**合并后**的权重计 208 条（`gateup` 合成 N=8704、`qkv` 合成 N=3584）。两者字节总数相同，只是计数单位不同。）

**与仓库现有文档不一致，需登记（原数错在哪）**：`docs/sm70_qpn_npad_design.md:115-125` 与 `docs/sm70_npad_ar_chunk_landing_plan.md:552` 写"侧车 **4.23 GB**"。该表第一行是「N=5120, K=5120, 128 条 = 1.887 GB」，张数 128 ✓（`down_proj` 64 + `out_proj` 48 + `o_proj` 16），但**这一组全部切输入维**，真实 `K_local` 是 4352 / 1536 / 1536，不是 5120。用真实 K 算这组只有 **1.085 GB**，原表高估 **0.802 GB**；而 4.230 − 3.430 = **0.800 GB** —— 差额全部来自这一行。**这是【我的推断】，不是实测**；要坐实只需在引擎里把 `Nvfp4QpnPackedBytes` 对全部侧车的返回值累加打印一次（现在没有这个日志，`fastllm-linear-fp8.cu:3638` 只分配不记账）。若按原表 4.23 GB，则两份常驻是 8.80 GB，本文其余结论方向不变、数字大约 +10%。


### B-2. 读盘量：FastLLM 单进程读一份 19.1 GiB，1Cat 4 个 worker 各读一份

【我的推断，但有源码直接支撑】

- FastLLM 是**单进程 4 卡**（`--tp 4` → `interpreted as using 4 CUDA device(s) => cuda:0,1,2,3`，`/tmp/b80k_c1_on.log` 第 2 行），加载器在进程内一次性把 19.147 GiB 全部读进主机内存，再切分+分发到 4 张卡。**磁盘只读一遍。**
- 1Cat 是 4 个独立 worker 进程（`Worker_TP0..TP3`，`/tmp/serve_e4m3_native.log:62-65`），每个 worker 各自跑一遍 `safe_open(...).get_tensor()`。关键点：**`get_tensor` 返回的是完整张量，切片发生在之后**——`sharded_weight_loader` 先 `get_tensor` 拿到全量，再 `narrow(shard_axis, ...)` 切自己那一份（`weight_utils.py:1489-1497`），`default_weight_loader` 要求 `param.size() == loaded_weight.size()`（`weight_utils.py:1455-1459`）。
- 所以 1Cat 的理论读盘量是 **4 × 19.147 GiB ≈ 76.6 GiB**，是 FastLLM 的 **4 倍**（4 个 worker 在同一时间读同一批文件）。这部分靠 page cache 兜住——第 2 个及之后的 worker 命中页缓存，实际只在第一次落盘。它们的 `Loading weights took` 也印证了这点：TP0 是 13.25 s，draft 是 2.71 s（`/tmp/serve_e4m3_native.log:98,128`）。
- **但这不构成 FastLLM 的优势**：page cache 把它变成内存拷贝，代价是每个 worker 进程各自多占一份主机内存。FastLLM 靠"单进程 + 按层流式"把主机峰值压到 8.78 GiB（`qwen35_streaming_load.md:34`），1Cat 4 个 worker 各自持有自己那份。**结论：两种拓扑的取舍不同，不是"谁更好"，但 FastLLM 在"读盘量 × 主机峰值"这个乘积上更省。**

---

## C. 可移植路线表

| 路线 | 预期收益（数字 + 算法） | 成立条件 | 主要风险 | 状态 | 证据等级 |
|---|---|---|---|---|---|
| **C-0 按层流式加载 / 降加载期主机峰值** | 主机 RSS 峰值：单卡 27B GGUF Q4_K_M **22.87 → 8.78 GiB**；单卡 9B BF16 **21.43 → 4.32 GiB**；TP2 FP8+DFlash2 **34.07 → 8.96 GiB**。代价：单卡启动 +0.6～3.9 s | 完整目标权重布局 + 纯 CUDA 设备映射（`qwen35_streaming_load.md:8`） | 已解决（回归含 CPU 源释放、字节一致、draft 隔离） | **已拿** | **【引自文档】** `docs/qwen35_streaming_load.md:29-52`，commit `a714ea6d` |
| **C-1 只保留一份打包布局（丢掉 native block-16）** | 省 **4.566 GB = 4.252 GiB / rank**（= native 那一份），占 16 GiB 卡的 **26.6%**。算法：`native 4.252 GiB + 侧车 3.194 GiB = 7.446 GiB`（16 GiB 卡的 46.5%），QPN2 侧车能同时兜住 decode 和 prefill 后，native 成为唯一冗余项 → 省 native。换算 KV 容量（页大小随 KV dtype 变）：4.252 GiB ÷ **1.05 MB/页 ≈ 4147 页 ≈ 531 k token**（FP8 KV，200K 配置，`pf180k_run.log:5`、`serve_rot.log:30`）；÷ **2.10 MB/页 ≈ 2073 页 ≈ 265 k token**（FP16 KV，8K 配置，`b80k_c1_on.log:27`） | 必须先把 1Cat 的 **QPN2-packed prefill 分派**搬过来（`nvfp4_qpn4_sm70.cu:663-678`），否则 M>32 的 prefill 没有 native 布局可用；FastLLM 现有 QPN2 明确是 decode-only（`fastllm-linear-fp8.cu:3681` `n < 1 || n > 32` 直接返回 false） | ① 移植量大（1Cat 那 1111 行含 QPN4 + packed prefill）；② prefill 侧已有"−1.43%"回退史（`sm70_nvfp4_coverage_plan.md:137` 要求 ±0.5% 门）；③ greedy sha256 门必须过 | **未做**（仓库里 QPN4 已被显式判"不进默认范围"：`sm70_status_and_backlog.md:128,469`；但"只搬 packed-prefill 分派、不搬 QPN4 decode"是**未被否掉**的子集） | **【我的推断】**（字节账来自源码 + checkpoint header；QPN4 判否来自**【引自文档】**） |
| **C-2 加载期不再把 E4M3 scale 展开成 FP32（省 host 侧一份 1.058 GiB/rank 的中间缓冲 + 一遍全量 CPU pass）** | 主机/设备各少 **1.136 GB/rank（1.058 GiB）** 的中间分配（4.252 − 3.194 = 1.058 GiB）；CPU 侧少一遍对 3.044 GB packed 数据的逐 block 改写。算法：12 B/16elem → 9 B/16elem | 与 C-1 是同一件事的两面：要让 native 保存原始 E4M3 字节，就得让所有消费 native 的 kernel 学会读 packed scale（TurboMind `PrepareNvfp4InPlace`、Marlin 都不会） | 动的是**全架构共用的布局契约**，不只 SM70；`sm70_status_and_backlog` 的"不做"清单里第 1 项精神上覆盖它 | **未做，且我倾向不做** | **【我的推断】** |
| **C-3 mmap 直读 + 零拷贝** | 上界 **0**。理由：① HF safetensors 路径本来就没走 mmap，开 `USE_MMAP` 也管不到它；② 即使管到了，`src/fastllm.cpp:2768-2774` 显示 mmap 路径 H2D 前会先 `memcpy` 进堆缓冲，本身也不是零拷贝；③ 数据最终必须过一次 PCIe 进 HBM，这是不可省的 19.1 GiB 搬运 | 需要"改完真的更快"的机制 | 改动散落在 `Data` / `ToDevice` / `FileMmap`，收益却是 0 | **不建议动** | **【我的推断】**（源码事实）+ **【引自文档】**（`qwen35_streaming_load.md` 已把主机峰值降下来，说明峰值不是 mmap 问题） |
| **C-4 加载期预读（POSIX_FADV_WILLNEED / 后台 prefetch 线程）** | 上界 ≈ 0。算法：FastLLM 的读序列是"最多 16 线程按字节切分全量顺序读一次"（`src/model.cpp:4890,4939-4953`），预读只能把同样的页更早拉进 page cache，**不能降低总 IO 量**；模型 19.147 GiB、本机 RAM 30 GB（`free -g`），page cache 装得下。对比：1Cat 的 auto-prefetch 只在网络 FS 且装得进 RAM 时开（`weight_utils.py:961-1005`），在本机 XFS 上它自己的日志就写着 disabled（`/tmp/serve_e4m3_native.log:89`）| 只在 checkpoint 超出 page cache、且读放大高（见 B-2）时才有意义 | 收益为 0 时是纯复杂度 | **不建议动** | **【我的推断】** + **【引自日志】** |
| **C-5 搬迁 tensorizer / modelexpress / runai_streamer 这类"加速加载器"** | 上界 ≈ 0。tensorizer 需要预先序列化成自己的 artifact（`tensorizer_loader.py:43-66`）；modelexpress 是 RDMA 远端权重服务（`modelexpress_loader.py:33-59`）；runai_streamer 是流式库（`runai_streamer_loader.py:21`）。三者解决的问题都是"网络/远端加载 + 冷缓存"，本机是单块 NVMe（`lsblk`: `nvme0n1 faspeed P8-256G`）、模型 19.1 GiB 装得进 30 GB RAM | 需要网络 FS 或超出 RAM 的 checkpoint | Python/PyTorch 生态的库，C++ FastLLM 无法直接复用；自己造一遍是几十天量级 | **不建议动** | **【我的推断】** + **【引自文档】**（`/tmp/serve_e4m3_native.log:89` 里 1Cat 自己都在日志里说 XFS 不是网络 FS、不开 prefetch） |
| **C-6 QPN2-packed prefill（只搬分派，不搬 QPN4 decode）** | 换个方向算：不是为 speed，而是 C-1 的**前提**。1Cat 用它在大 M 上省掉 native 副本 | 需要 `M >= min_prefill_m` 的 packed GEMM（1Cat 默认阈值 1024，`vllm/envs.py:1777`） | 与 `sm70_4x200k_rotate_plan.md:166-175` 的结论有张力：那边实测 prefill 本来就跑在 dequant+cuBLAS 上、`FASTLLM_NVFP4_PREFILL_CUBLAS` 开着也没变化，所以"打包能救 prefill"这件事在 FastLLM 侧**没有实测支撑** | **未做** | **【引自文档】**（1Cat 侧实现与阈值）+ **【我的推断】**（对本项目的收益） |

---

## D. 一句话结论

**加载侧已经没有可捡的收益，不建议为"加载"本身再立项。** 理由链：

1. **这块最大的收益已经拿掉了。** 按层流式加载在 2026-09-08 落地（`a714ea6d`），主机 RSS 峰值单卡 27B GGUF **22.87 → 8.78 GiB**、TP2 FP8+DFlash2 **34.07 → 8.96 GiB**（**【引自文档】** `docs/qwen35_streaming_load.md:29-52`）。这不是待做项，是【已拿】。
2. **"加载即重排"这条不存在可捡的空间。** 两边都只在加载/warmup 阶段重排一次：FastLLM 在 warmup 后一次性建全部 QPN2 侧车（`src/models/qwen3_5.cpp:9685-9720`），1Cat 在 `process_weights_after_loading` 里做（`base_loader.py:80`）。FastLLM 还有 `nvfp4Qpn2Wanted` / `IsRepacked` 两个粘性门挡住重复 repack（`fastllm-linear-fp8.cu:3595`、`:1445-1485`）。**前向期没有重复转换。**
3. **加载耗时不是瓶颈。** FastLLM 进程起→可服务约 **41.9 s**（**【引自日志】** `/tmp/serve_rot.log:1` 22:18:14.985 → `:55` 22:18:56.901，该 run 无 MTP/无 draft）。1Cat 同机同模型是加载 20.5 s + 引擎初始化 62.0 s ≈ 82.5 s（**【引自日志】** `/tmp/serve_e4m3_native.log:131,238`）。**但这两个配置不可直接比**：1Cat 那跑开了 DFlash2 投机 + `torch.compile`（编译单独占 46.76 s），FastLLM 那跑没开。所以只能说"FastLLM 的加载侧不比 1Cat 差"，不能说快 2 倍。
4. **唯一真正剩下、并且有数字的，是每卡同时常驻两份 NVFP4 布局。** native block-16 **4.566 GB** + QPN2 侧车 **3.430 GB** = **7.996 GB/rank**，占 16 GiB 卡的 **46.5%**；只留侧车能省 **4.566 GB = 26.6%** 显存（**【我的推断】**，算法与交叉校验见 B-1）。**但它换的是显存不是时间**，而且前提是搬 1Cat 的 QPN2-packed prefill 分派（`nvfp4_qpn4_sm70.cu:663-678`）——那属于算子覆盖，不属于加载模块；QPN4 那一整块已被显式判"不进默认范围"（`sm70_status_and_backlog.md:128,469`），且 `sm70_4x200k_rotate_plan.md:166-175` 实测表明 FastLLM 侧 prefill 本来就跑在 dequant+cuBLAS 上、"打包能救 prefill"没有本地证据。
5. **mmap / 预读 / 加速加载器三条的上界都是 0**，理由逐条写在 C-3 / C-4 / C-5：HF 路径不走 mmap 且数据必须过一次 PCIe；顺序全量读无法靠预读减少 IO 量；tensorizer/modelexpress/runai_streamer 解决的是网络 FS 或超 RAM checkpoint，本机是单块 NVMe + 30 GB RAM。

**结论：照现状不动。** 若确实要动显存，正确的立项口是"NVFP4 单布局化（含 packed prefill）"，应该归到 `sm70_nvfp4_coverage_plan.md` 的算子覆盖线，而不是加载模块线；且按该文档的既定顺序排在 `lm_head` 之后。

