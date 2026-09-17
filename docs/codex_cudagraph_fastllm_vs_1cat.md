# FastLLM vs 1Cat-vLLM 1.5.0：计算图 / CUDA Graph 对照

**方法说明**：本报告**只读代码 + 只读已有文档/审计记录**，没有跑任何 benchmark、编译或测试。
因此绝大多数条目是【引自文档】或【我的推断】，只有明确标注的才是已有实测。
目标场景：SM70/V100、4 卡 TP4、Qwen3.8-27B-QUASAR-NVFP4，长上下文 prefill（180K，chunk 4096）与 decode。

**三条先决事实（本轮已核实，全部来自代码/文档，非我新测）**

1. 设备侧并发恒为 1，且这是**引擎在 SM70 上的代码结构决定的**，不是负载特性。
   【引自文档】`docs/sm70_4x200k_rotate_plan.md:383-384`：180K trace 四卡各自 `max_concurrent=1`、
   `>=2 内核在飞的时间 = 0.0000 s`。旁证 `/home/nsys/pf180k.sqlite` 存在（1.26 GB）。
2. 降主机阻塞 ≠ 拿收益。【引自文档】`.audit/sm70-longctx-kv.tsv` 第 85 行（2026-09-16T08:52:30Z）：
   `FASTLLM_SM70_NCCL_ASYNC=1` 让同步 23120→12800（−45%）、阻塞 152.87→82.06 s（−46%），
   但**并发仍 1、`t_>=2` 仍 0.0000 s、墙钟 38.1410→38.2326 s 不变**（sha256 相同）。
   同一 tsv 第 84 行还撤掉了此前"降 92%"的错误数字（那是两个不同负载相除）。
   **→ 任何"图能减少 host gap"的说法，必须证明它改变墙钟，否则不算收益。**
3. SM70 上 `FastllmCudaGetNcclForceSync()` 恒为 true（真正的门在 `src/models/basellm.cpp:4099` 的
   `if (FastllmCudaRuntimeArch() >= 75 || sm70Override) FastllmCudaSetNcclForceSync(false);`，
   SM70=70 不满足；`docs/sm70_4x200k_rotate_plan.md:389-390` 与父 agent 转述的 `:4093` 是那段注释的行号，
   不是判断语句本身），每次集合通信后跟一次主机
   `cudaStreamSynchronize`（`src/devices/multicuda/fastllm-multicuda.cu:2450` +
   `:3210` 及三个兄弟集合通信）。【代码核实】

**关于第 3 条的修正（本轮新发现，重要）**：这条门**不会**挡住图捕获。
`FastllmNcclPostSyncEnabled()`（`fastllm-multicuda.cu:2445-2464`）在返回前查
`cudaStreamIsCapturing`，**只要处于捕获态就返回 false**，注释原文："流捕获期间必须返回 false：
`cudaStreamSynchronize` 属于捕获期间被禁止的调用，会直接 invalidate 正在进行的 CUDA graph 捕获"。
同一文件 `:3164-3173` 的 `waitForRanks` 也在捕获态把 `rendezvous` 置空来绕过主机会合；
`include/devices/multicuda/ncclsubmitrendezvous.h:12-13` 写明"Capture must bypass BOTH boundaries;
replay does not execute this host code at all"。**→ 捕获期与回放期都不受这条门影响；
这条门只在 eager 路径上生效。所以"per-collective 主机同步从根上挡住 prefill 成图"这个假设不成立。**

---

## A. 两份实现的计算图要点

### A.1 FastLLM（`/home/fastllm`）

1. **捕获 API 是薄封装，只有 begin/end/instantiate 三个动作。**
   `src/devices/cuda/fastllm-cuda.cu:732` `FastllmCudaGraphBeginCapture()` →
   `cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)`（`:745`）；
   `:1319` `FastllmCudaGraphEndCapture()` → `cudaStreamEndCapture`；`:1334` `FastllmCudaGraphInstantiate()`。
   注意模式是 **ThreadLocal** 而非 Global——这是 4 卡 TP 能各自独立捕获的前提。
2. **图池是"清账式"的：捕获期记账，finalize 期 pin 指针。**
   `:137-152` 定义三态 `Enum FastllmCudaGraphPoolPhase {IDLE, CAPTURING, FINALIZING}` +
   `std::set<void*> fastllmCudaGraphPoolTouchedDuringCapture`。
   `:5057` `FastllmCudaGraphMemoryPoolBegin()` 清表并置 CAPTURING；
   `:5071` `FastllmCudaGraphMemoryPoolEnd(reservedPointers)` 把捕获期碰过的池地址逐个
   `buffer.graphPins++` 并交给调用方持有；`:5137` `Abort()` 丢弃；`:5145` `Release()` 逐个 `graphPins--`。
   目的是：图保留的是**裸设备指针**，而 host 侧 `Data` 临时对象捕后即析构，池不能让这些地址被复用。
3. **捕获期禁止什么：malloc / free / host sync 三者都禁，且失败要跨 rank 一致。**
   - 禁真实 `cudaMalloc`：`:5038` `FastllmCudaRetryMallocAfterReleasingIdle()` 开头
     `if (FastllmCudaGraphIsCapturingFast()) { FastllmCudaSetThreadError(); return false; }`，
     注释"cudaFree is forbidden while a stream is being captured"。
   - 因此必须在捕获**之前**备好"分配失败占位地址"：`:71` `FastllmCudaGraphPrepareAllocationFailurePlaceholder()`
     在 `:732` 的 `BeginCapture` 里、`cudaStreamBeginCapture` **之前**调用（`:736-741`），
     按设备各留一块 256 B 进程生命期缓冲（`:88-93`）。
   - 失败语义是"让失败的 rank 走到共同的 abort barrier"，而不是单 rank 解栈：`:59-63` 注释明说，
     否则一个 rank 退出时其它 rank 还在录 NCCL 集合通信。
   - 禁 host sync 由 NCCL 侧负责（B 表末行 + C 表后说明）。
   - 失效检测两层：线程错误标志 + 捕获纪元聚合 `fastllmCudaGraphErrorFlag`（`:56-58`、`:103`、`:122`），
     外加 `:765` `FastllmCudaGraphCaptureInvalidated()` 兜底（注释点明 FlashInfer、独立 kernel 的 printf
     这类不走 `showError` 的失效）。
4. **有"捕获后重写拓扑"的能力——1Cat 侧没有这一项。**
   `:1000+` `FastllmCudaGraphOptimizeQwen35Moe()`：`cudaGraphGetNodes` 取全部节点，
   按 marker kernel（`:823-841` 四个空 kernel：fork/sharedDone/routedBegin/join）识别分支区间，
   再 `cudaGraphAddDependencies` 把原本串行的 MoE 拓扑改成并行，最后 `cudaGraphDestroyNode` 删 marker。
   安全前提显式检查：`:1044-1055` 若图中含 `cudaGraphNodeTypeMemAlloc/MemFree`，
   直接返回 `FASTLLM_CUDA_GRAPH_MOE_RECAPTURE_WITHOUT_MARKERS` 放弃优化（CUDA 禁止在这种图里销毁节点）。
   模型侧还有一道事后回退：`FastllmCudaMergeMOEUsedGraphUnsafeFallback()` 在捕获体前后被检查
   （`src/models/qwen3_5.cpp:12248-12251`），一旦 MoE 用了图不安全路径，整次捕获作废退回 eager。
5. **指针表（pointer table）是"owner 作用域 + 强制静态地址"机制。**
   `:16787` `struct FastllmCudaGraphPointerTableScope`、`:16796` `...Begin(owner)`、
   `:16802` `...End()`；`:16843` 起在分配/更新路径判断 `graphScoped`。
   调用点是 `src/models/qwen3_5.cpp:3868-3874` 的 RAII 包装，在 `:16422`（eager）与
   `:16589-16590`（replay）各实例化一次——即**捕获取的与回放用的必须是同一批地址**，
   这也是 `:11110`、`:11127` 那些
   `AssertInFastLLM(... "requires aligned paged cache layout/positions across attention layers")` 的由来。
6. **模型侧图条件是"窄门"，只对 decode 开；`--low_gpu_mem` 与图完全解耦。**
   `src/models/qwen3_5.cpp:10719-10724`：
   ```cpp
   if (!Qwen35CudaGraphEnabled() || batch <= 0 || batch > maxCudaGraphDecodeBatch ||
       !all1 || isPrefill || ... || seqLens[0] != 1 || ...) return finishGraphEligibility(false);
   ```
   其中 `all1 = (每个 seqLens[i]==1)`、`isPrefill = !all1`（`:15305-15308`）。
   **`isPrefill` 为真直接 return false——prefill 全走 eager。**
   图开关本身走 `tools/fastllm_pytools/util.py:320-343` `_cuda_graph_auto_supported()`，
   门限常量已是 **70**（`:317`），Volta 另留显式 opt-out `FASTLLM_QWEN35_SM70_CUDA_GRAPH=0`（`:339-343`）。
   而 `--low_gpu_mem` 只置 `cuda_embedding=False` 与 `GPU_TOKEN_HANDOFF=0`，
   并明确打印 "CUDA graph selection is unchanged"（`:601-608`）。
7. **解码图按"精确 batch"建索引，不做向上取整。**
   `:3970` `GetQwen35CudaGraphDecodeState(model, gpuId, batch)`，key = `std::make_tuple(model, gpuId, batch)`（`:3973`）。
   我 grep 过 `qwen3_5.cpp` 全文件，**没有任何 padTo/roundUp/`std::upper_bound` 之类"把 batch 抬到最近档位"的逻辑**
   （`grep -n "padTo|padBatch|roundUpBatch|paddedBatch"` 零命中）→ 没捕过的 batch 号只能落回 eager。
8. **多档位预捕获已实现，策略与 1Cat 同类。**
   `:4274` `Qwen35MaxCudaGraphDecodeBatch()`（默认 32、硬上限 128，
   env `FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH`；**MTP 打开时直接返回 1**，见 `:4245-4252`）；
   `:4301` `Qwen35PreCaptureBatches()` 取几何序列（1,2,4,8…）+ 端点，注释原文：
   "Match the graph-size policy used by high-throughput engines: capture geometrically
   spaced shapes, plus explicit endpoints."
   两个开关：`:4281` `Qwen35PreCaptureStartBatch`（下界）、
   `:4295` `Qwen35DensePreCaptureBatches`（改为 1..max 密集）。
   驱动点在 `:10346` 的 warmup 循环，**倒序**捕获（`:10347-10349`，先最大后小，让池复用）。
9. **预捕获 warmup 跑"8-token prefill + 2 步 decode"，但只有 decode 那部分真成图。**
   `:10403` `const int cudaGraphPreCapturePrefillTokens = 8;` → `:10422` `ForwardGPU(...)` 跑一次 8-token prefill
   → `:10428-10429` `cudaGraphPreCaptureDecodeSteps = 2` 跑 2 步 decode。
   由第 6 条，那一次 8-token prefill 在 `isPrefill` 检查处就被拒（`all1=false`），只能走 eager；
   真正成图的是紧接的 2 步 decode。**所以第 8 条那些档位全部是 decode 图。**
   旁证：`:10539` 打印 `"Qwen3.5 serving eager prefill warmup"`、
   `:10546` 注释 `"Graph capture can pin eager-prefill scratch"`、
   `:10560` `servingPrefillTokenLimit = GetChunkedPrefillSize()`、
   `:10576-10577` 调用 `runEagerPrefillWarmup(..., "batched serving"/"ragged batched serving")`
   ——**prefill 的 scratch 是按 chunk 在 eager 下预热的**（chunk 默认 2048，见 `:8839`
   `defaultChunkedPrefillSize = 2048`）。
10. **`src/graph.cpp` 与上面这套 CUDA Graph 无关；另有两条独立小图与内建自检。**
    `graph.cpp` 是 `ComputeGraph` 解释器：`OptimizeComputeGraph()`（`:6`，合并同输入连续 Linear +
    融合 Swiglu）、`RunComputeGraph()`（`:126`，逐 op 调 `excutor.Run(op.type, ...)`，`:617`）。
    消费方只有 `GraphLLMModel` 系列（`src/models/graphllm.cpp:139,288`），下挂 `src/models/graph/` 的
    qwen2/gemma2/phi3/minicpm3/telechat。**我 grep 过 `qwen3_5.cpp` / `qwen4_exp.cpp` / `deepseekv4.cpp`，
    零命中 `ComputeGraph`**：这三个现代模型走 `ForwardSingleGPU*` 专用实现，不经这条解释器。
    两条小图：采样侧 `fastllm-cuda.cu:1348` `FastllmCudaTensorParallelGreedyGatherGraphCreate`（手工建图 +
    per-rank peer 拷贝节点）；自定义 all-reduce 的图缓存 `fastllm-custom-allreduce.cu:1731`（构建）、
    `:2016-2043`（`customGraphs` / `ncclGraphs` 两套 exec）。
    自检：`fastllm-cuda.cu:1240-1310` 报 `FastllmCudaGraphQwen35MoeParallelSelfTest` 与
    `...AllocationFallbackSelfTest`。

### A.2 1Cat-vLLM 1.5.0（`/tmp/1Cat-vLLM-1.5.0`）

1. **档位集合是"显式配置 + 自动生成"两路。**
   `vllm/config/compilation.py:640` `cudagraph_capture_sizes`；自动生成规则见 `:692-693` 文档字符串
   `[1,2,4] + range(8,256,8) + range(256, max+1,16)`，`max = min(max_num_seqs*2, 512)`（`:698-700`）；
   实现于 `vllm/config/vllm.py:2410-2411`，并把 `max_num_batched_tokens` 也塞进列表（`:2520-2525`）。
2. **SM70 有专门一套档位，比通用规则窄得多。**
   `vllm/config/vllm.py:83` `_SM70_NOMTP_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16)`；
   `:164` `_sm70_nomtp_cudagraph_capture_sizes(max_num_seqs)` 取 `min(max_num_seqs,16)`，
   再 `update((1,2,max_graph_reqs))` 后排序。MTP 打开时换成 `:173` `_sm70_mtp_cudagraph_capture_sizes()`，
   基于 `:84` `_SM70_MTP_CUDAGRAPH_REQUEST_SIZES = (1,2,4,6,8,12,16)` 乘 `decode_query_len`。
3. **padding 到档位是一张预计算查表，语义是"向上取整到最近档位"。**
   `vllm/v1/cudagraph_dispatcher.py:291` `_compute_bs_to_padded_graph_size()`：
   对每对 `(start,end)` 填表，`bs==start` 时填自己、否则填 `end`（`:304-310`）；
   `:348` `_create_padded_batch_descriptor()` 用它把 `num_tokens` 变 `num_tokens_padded`。
4. **不命中档位的退化路径是 eager（NONE），不是回落小图。**
   `:614-624` `dispatch()` 早退条件含 `num_tokens > max_size` → 返回 `(NONE, BatchDescriptor(num_tokens))`；
   两个 mode 都没命中时 `:711` 同样 `return CUDAGraphMode.NONE, ...`。
5. **图记忆池是"全局单池 + 描述符 → pool 映射"。**
   `vllm/compilation/cuda_graph.py:269` `_graph_pool_for_descriptor()`；`:290` 用
   `forward_context.batch_descriptor` 查表；`:352` 取 `graph_pool`；pynccl 侧用 `set_graph_pool_id` 绑定（`:20`）。
   条目容器 `concrete_cudagraph_entries[batch_descriptor]`（`:305-312`），
   `CUDAGraphEntry` 定义见 `:128-140`（含 DEBUG 用的 `input_addresses`）。
6. **入口在 model_runner 侧批量捕获，且大形状先捕。**
   `vllm/v1/worker/gpu_model_runner.py:1832` 建 `CudagraphDispatcher`；`:11721` `capture_model()`；
   `:11772-11774` 遍历 `get_capture_descs()` 逐 mode 批量捕；
   `:11768-11770` 注释明说 "Capture the large shapes first so that the smaller shapes can reuse
   the memory pool"。`get_capture_descs()` 在 dispatcher `:714-744`，顺序 **PIECEWISE 先、FULL 后**、
   组内按 `num_tokens` 降序。
7. **"可打断图"（breakable）用运行时流捕获断点替代编译期切图。**
   `vllm/compilation/breakable_cudagraph.py:3-19` 文档：不做 FX 切分，而是
   "a single capture context drives the whole forward and intercepts attention / kv-cache custom ops
   at the dispatcher to end the current stream capture, run the op eagerly, and resume capture"。
   核心 `:126` `class BreakableCUDAGraphCapture`：`_begin_segment()`（`:176`）`g.capture_begin(pool=...)`、
   `_end_segment()`（`:186`）`capture_end()` 并 `append(self._current_graph.replay)`、
   `:199` `add_eager(fn)` 做"结束本段 → 跑 eager → 记 segments → 开新段"，
   产物是 `:204` `replay()` 顺序执行的一串 callable。
   打断点由算子级装饰器标注，不靠字符串列表：`:60` `eager_break_during_capture`，
   `:95-116` 在捕获态转 `capture.add_eager(...)`；标注点见
   `vllm/model_executor/layers/attention/mla_attention.py:1051`、
   `vllm/models/deepseek_v4/attention.py:646`、`layers/sparse_attn_indexer.py:85`、
   `sparse_attn_indexer_kpool.py:245`、`layers/quantization/fp8_sm70_moe.py:340`、
   `models/glm5next/nvidia/kda.py:434`。
8. **breakable 对 prefill 与 decode 一视同仁，且不依赖 torch.compile。**
   `breakable_cudagraph.py:337-341` 注释："Unlike the original CUDAGraphWrapper which strictly matches
   a single runtime_mode, this wrapper captures whatever the dispatcher emits (any non-NONE runtime_mode)
   -- **breakable's capture is identical for prefill and decode**"。开关是 env
   `VLLM_USE_BREAKABLE_CUDAGRAPH`（`vllm/envs.py:620`、`:1232-1234`，默认 False）。
9. **SM70 默认策略是 `FULL_AND_PIECEWISE`，且 prefill 的 key 被刻意放宽。**
   `vllm/config/vllm.py:1620-1648`：`sm70_flash_0dot3_compile_graph` 为真、`is_device_capability((7,0))`
   且 `VLLM_SM70_FLASH_ATTN_V100` 时设 `mode=VLLM_COMPILE`、`cudagraph_mode=FULL_AND_PIECEWISE`；
   `:1806-1809` 日志字符串写明。语义见 `vllm/config/compilation.py:622-623`：
   "Capture full cudagraph for decode batches and piecewise cudagraph for prefills"
   → **1Cat 在 SM70 上 prefill 也进图（piecewise 形式），这是与 FastLLM 最实质的差别。**
   为让可变长 prefill 能成图，`cudagraph_dispatcher.py:554-557` 在 `mixed_mode()==PIECEWISE` 时把
   descriptor 替换成 `num_reqs=None, uniform=False`（注释：FULL 需要精确 `num_reqs`，因 FA3 的
   scheduler_metadata 依赖它；PIECEWISE 不需要），从而把"prefill 长度可变"归约成"只按 num_tokens 档位"。
10. **还有按 KV 上下文长度再分档的"上下文桶"，以及捕获期全局闸与 DEBUG 断言。**
    `vllm/v1/cudagraph_dispatcher.py:22-24` `_SM70_FP8_KV_BATCH_CONTEXT_SIZES = (4,8,16)`、
    `_SM70_FP8_KV_LONG_CONTEXT_MIN_SEQ_LEN = 16384`、`_SM70_E4M3_B1_WAVE_LONG_CONTEXT_MIN_SEQ_LEN = 49152`；
    `:60-89` `_get_sm70_dsv4_decode_context_buckets()` 按 `bucket_multipliers = (1,2,8,32,64)` 生成；
    `:215-225` `_add_context_bucket_keys()`；dispatch 用 `:696` `_get_context_bucket_descriptor` 把
    `attention_context_len` 折进 key → **图宽度不只按 batch，还按 KV 长度分档**，FastLLM 侧无对应物。
    捕获期全局闸：`vllm/compilation/monitor.py` 的 `validate_cudagraph_capturing_enabled()`，
    在 `cuda_graph.py` 与 `breakable_cudagraph.py:365` 调用；`gpu_model_runner.py:11764`
    `set_cudagraph_capturing_enabled(True)`、`:11785` 置回（注释："any unexpected cudagraph capturing
    will be detected and raise an error after here"）。
    回放期地址一致性断言：`breakable_cudagraph.py:410-418`、`cuda_graph.py:158-161`（仅 DEBUG 开）。
    上游参考 `/tmp/vllm_cudagraph_utils.py` 是这套的三层结构：`:137` `class CudaGraphManager`、
    `:501` `class ModelCudaGraphManager`、`:59` `BatchExecutionDescriptor`、`:98` `_is_compatible`；
    `:761` `profile_cudagraph_memory()` / `:866` `_extrapolate_full_graph_memory()` 做显存采样与外推。

## B. 差异清单表

| 维度 | FastLLM 现状 | 1Cat 做法 | 差距是否要紧 |
|---|---|---|---|
| **允许的动态性：档位集合** | 几何序列 1,2,4,8…+端点，默认 max=32/硬上限 128（`qwen3_5.cpp:4301`、`:4244`）；MTP 开时**只留 batch=1**（`:4252`） | SM70 默认档位 `(1,2,4,8,16)`（`vllm.py:83`、`:164`）；通用规则 `[1,2,4]+8..256步8+256..步16`（`compilation.py:692`） | **不要紧**。本场景 Qwen3.8-27B-QUASAR-NVFP4 目标 max_batch=4、实际并发 C≤4，两边的档位都够覆盖 |
| **不命中档位怎么退化** | 精确 batch 建 key（`:3970-3973`），**无向上取整**，未捕过 → 整步 eager | 预计算 `_bs_to_padded_graph_size` 表，向上取整到**最近档位**（`cudagraph_dispatcher.py:291-310`） | **要紧（实际影响最大的一条）**。见 C 表 L1：FastLLM 在"非档位 batch"上会整步退回 eager，而不是用邻近的稍宽图 |
| **档位是否含 batch 之外的轴** | 只有 batch（key = `(model, gpuId, batch)`，`:3973`） | batch + **KV 上下文桶**（`cudagraph_dispatcher.py:22-24`、`:60-89`、`:215-225`）+ LoRA 数（`:540`） | **中等要紧**。长上下文 decode 的 attention 内核路由可能随 KV 长度变（`attention/paged/fastllm-paged-attention-native.cu:3194-3200` 按 dtype/GQA 分路），但 FastLLM 走的是同一张图内条件分支，不是不同图；是否真需要分图**未验证** |
| **prefill 是否成图** | **否**。`isPrefill` 直接 return false（`:10721`）；prefill scratch 走 eager 预热（`:10539`、`:10546`、`:10576-10577`） | **是**。SM70 默认 `FULL_AND_PIECEWISE`（`vllm.py:1620-1648`），`compilation.py:622-623` 明说 piecewise 覆盖 prefill；key 放宽到 `num_reqs=None, uniform=False`（`cudagraph_dispatcher.py:554-557`） | **要紧，但收益不明**。1Cat 侧的机制已核实；收益需实测，见 C 表 L2 与 D |
| **不可捕获段怎么切开** | 无此机制。整步图要么全捕要么不捕；靠 `FastllmCudaMergeMOEUsedGraphUnsafeFallback()` 事后作废（`qwen3_5.cpp:12248`） | breakable：运行时 `capture_end()` → eager 跑 → `capture_begin()` 续捕（`breakable_cudagraph.py:176-206`），打断点用 `@eager_break_during_capture` 标在算子级（`:60`） | **要紧（架构级）**。这是"能不能把 attention/通信留在 eager 之外、只把 GEMM 段烤进图"的根本能力；FastLLM 只有"全或无" |
| **捕获后能否改拓扑** | **能**。`FastllmCudaGraphOptimizeQwen35Moe`（`fastllm-cuda.cu:1000+`）用 marker kernel 重排 MoE 分支为并行，并检测 alloc/free 节点后安全放弃（`:1044-1055`） | 未见等价物（未找到：breakable 只在段间切换 eager，不重写段内依赖） | **不要紧**（FastLLM 反而领先这一项） |
| **捕获期显存记账** | 显式三态池 + `graphPins` 计数（`:137-152`、`:5071-5130`、`:5145+`）；失败时用 256 B 占位地址把 rank 带到共同 abort barrier（`:59-93`） | 全局 graph pool + 描述符映射（`cuda_graph.py:269-352`）；torch 侧 `graph_capture()` 上下文（`gpu_model_runner.py:11759`）；另有显存外推估计 `profile_cudagraph_memory`（utils `:761`、`:866`） | **不要紧**（两边都完备，实现风格不同） |
| **捕获 vs 回放的开销收益** | 已有实测：8K decode 开图 58.57 → 71.89 tok/s（**+22.7%**）【引自文档】`docs/sm70_1cat_port_plan.md:54`；关图 17.07 → 13.91 ms/token（**约 −20%**）【引自文档】`.audit/sm70-longctx-kv.tsv:22` | 文档口径为"graph 是 14 ms 的前提"、图内每 rank 每 token 约 1062–1141 个 kernel【引自文档】`docs/sm70_qwen38nvfp4_1cat_gap_audit.md:57,102` | **两边收益量级已被证明同阶**，且 FastLLM 8K C=1 已达 75.13 tok/s 对 1Cat ~71【引自文档】`docs/sm70_1cat_port_plan.md:63-66` |
| **正确性保障：指针表** | `FastllmCudaGraphPointerTableScope`（`fastllm-cuda.cu:16787-16807`），模型侧 RAII 于 `qwen3_5.cpp:3868-3874`；配合大量 `AssertInFastLLM(... "requires aligned paged cache layout/positions")`（`:11110`、`:11127-11131`） | 池引用 + DEBUG 期输入地址断言（`breakable_cudagraph.py:410-418`、`cuda_graph.py:158-161`）；默认**不**校验地址（只在 DEBUG） | **不要紧**（FastLLM 更严） |
| **正确性保障：跨 rank 一致** | 显式 TP 屏障 `Qwen35TpDecodeGraphContext` + `Qwen35CudaGraphBarrier`（`qwen3_5.cpp:3667-3778`），`All/AllEqual/Max` 做跨 rank 表决，`BeginMemoryPool/EndMemoryPool` 只在 rank0 执行；`:10921-10926` 用 `AllEqual(localStatePhase)` 阻止"一 rank 回放、另一 rank eager" | 依赖 vLLM 的 collective_rpc / TP 同步约定，未见等价显式状态表决 | **不要紧**（FastLLM 更严） |
| **捕获期禁止什么（cudaMalloc/free/sync）** | 三者都禁：`cudaMalloc` 走占位地址（`:88-93`）；`cudaFree` 在 `:5038` 直接置错；`cudaStreamSynchronize` 由 `fastllm-multicuda.cu:2445-2464` 捕获态返回 false 规避 | torch `graph_capture()` 上下文 + `set_cudagraph_capturing_enabled` 全局闸；捕获前 `gc.collect()` + `empty_cache()` + offloader 同步（`breakable_cudagraph.py:376-384`） | **不要紧** |
| **SM70 上集合通信与图的关系** | 捕获/回放**都不执行**主机 `cudaStreamSynchronize`（`fastllm-multicuda.cu:2445-2464`、`:3164-3173`；`ncclsubmitrendezvous.h:12-13`）；门只在 eager 生效 | 未见对应的"per-collective 主机同步"设计 | **不要紧 —— 且这是本轮的关键澄清**：父 agent 假设的"per-collective 同步从根上挡住 prefill 成图"**不成立**，代码已显式规避 |
| **`--low_gpu_mem` 与图** | 与图选择解耦，只关 `cuda_embedding` 和 GPU token handoff，打印 "CUDA graph selection is unchanged"（`util.py:601-608`） | 无同名概念（vLLM 用 `gpu_memory_utilization` + `kv_cache_config.num_blocks`；`:1465-1479` 有 Mamba 相关 raise） | **不要紧** |

---

## C. 可移植路线表

所有收益一律标注算法与假设。凡我没有实测的，绝不写"实测"。

| 路线 | 预期收益（数字 + 算法） | 成立条件 | 主要风险 | 状态 | 证据等级 |
|---|---|---|---|---|---|
| **L1. 给 decode 图加"向上取整到最近档位"** | 省下的是"非档位 batch 的整步 eager 退避"。**算不出确定值**：需要 trace 里 `batch` 的分布。上界算法 `= Σ_{b∉档位} (每步图内时间 − 每步 eager 时间) × 步数`。以 8K 关/开图 17.07/13.91 ms/token 为单步差（**−3.16 ms**，【引自文档】`.audit/…tsv:22`），若并发 C=3（档位 1,2,4,8… 命中）则差为 0；只有出现 C=5,6,7,9… 或 MTP 场景（档位退化为 1，见 `:4252`）才吃这笔 | 档位集合与目标并发分布错位；且需要"用更宽的图跑更窄的 batch"在数值上安全（注意力 mask/位置编码要能容纳 padding 行） | 数值正确性（padding 行的 KV 页分配与 `AssertInFastLLM` 的对齐断言 `:11110`、`:11127-11131` 直接相关）；显存（每档保留一套 `reservedPointers`，见 `:4346-4348` 的记账公式 `64MB + maxBatch*8MB + graphCount*16MB`） | **未做**（代码里确认无 pad-up 逻辑） | 【我的推断】，单步差来自【引自文档】 |
| **L2. 把真实 chunk prefill 做成固定形状图** | **算不出确值，且很可能接近 0。** 唯一有引擎支撑的空间：180K 单条 prefill 104.2 s 中，稳态段设备空闲 **3.17 s（约 3%）**，且其中 >10 ms 的 44 个间隙里 **43 个**是 NCCL 内部 allreduce→broadcast 交接（每次约 34.5 ms），**不是等主机**。算法：收益 ≤ 稳态设备空闲 3.17 s，且这 3.17 s 里可被"消除 host gap"解释的部分预估 <0.5 s | 需要 (a) chunk 形状真固定、(b) 入图后不引入新的池未命中、(c) attention/通信可留在段外 | **（1）chunk 形状并非天然恒定**：`GetChunkedPrefillSize()`（`:10104-10115`）会被 `Qwen35LinearPrefixSnapshotIntervalTokens()` 按 pageLen 向下取整改写；`basellm.cpp:3665-3671` 默认 2048（`:8839`）；文档实测跑的是 chunk 4096（`docs/sm70_4x200k_rotate_plan.md:119`）——即**由配置决定，不是硬编码**。末块长度 `curLen = min(chunkSize, totalLen-st)`（`:32469`）也可变。**（2）更大的障碍**：`qwen3_5.cpp:11000+` 的图内分页缓存记账要求跨层 page 索引逐层对齐（`:11110`、`:11127-11131` 双重 `AssertInFastLLM`），而 prefill 每 chunk 都要**新分配** KV 页（`needNewPageHost`，`:10845-10853`）——这与"图保留裸指针不变"直接冲突。180K / chunk 4096 = 44 个 chunk，每 chunk 都要重新分配页 | **未做**；且经评估为**不建议做** | 【引自文档】（104.2 s / 3.17 s / 43-of-44）+【我的推断】（chunk 可变、页分配冲突） |
| **L3. 移植 breakable cudagraph（段间切 eager）** | **算不出，需实测。** 它是 L1/L2 的**前置能力**而非独立收益：只有它能同时满足"GEMM 段入图"与"attention/通信留 eager"。单独看，若沿用 FastLLM 现有"全图"语义则收益 0（现状已是全 decode 图） | 需在 FastLLM 里引入 (a) 算子级"此处断开"标注、(b) 段列表回放、(c) 每段各自的指针稳定性契约。工程量对应 `breakable_cudagraph.py:126-206` + `:60-116` 的等价物 | 与现有 `FastllmCudaGraphMemoryPoolBegin/End` 的单事务语义冲突（`:5057-5135` 设计为**每次捕获一个** transaction；多段会变成多次 begin/end）；与 `Qwen35TpDecodeGraphContext`（`:3667-3778`）的跨 rank 表决也要按段重做 | **未做** | 【我的推断】 |
| **L4. 把 attention 拆出图、只留 GEMM 段（L3 的具体用法）** | **收益 ≈ 0，已由实测否掉。** 依据：设备侧 `max_concurrent=1`、`t_>=2 = 0.0000 s`（【引自文档】`docs/sm70_4x200k_rotate_plan.md:383-384`），且 SM70 上每次集合通信后的 eager 主机同步（`basellm.cpp:4099`）本就阻止重叠；`FASTLLM_SM70_NCCL_ASYNC` 实验把同步降 45% 后**墙钟仍不动**（【引自文档】`.audit/sm70-longctx-kv.tsv:85`）。把算子移出图**只会增加** host 发射次数 | 需要先出现 `max_concurrent>=2` | 会反向增加 launch 税 | **已否掉/已实测** | 【引自文档】 |
| **L5. 用图降低 launch 税（现有 decode 图）** | 已拿到：8K decode **+22.7%**（58.57→71.89 tok/s）【引自文档】`docs/sm70_1cat_port_plan.md:54`；等效口径关图 17.07→13.91 ms/token（**−3.16 ms/token**，约 −20%）【引自文档】`.audit/…tsv:22`。1Cat 侧同一现象：图内每 rank 每 token 1062–1141 kernel【引自文档】`docs/sm70_qwen38nvfp4_1cat_gap_audit.md:57` | 已成立 | — | **已拿（默认开）** | 【引自文档】 |
| **L6. 长上下文 KV 桶（照 1Cat 的 context bucket 分图）** | **算不出，需实测**。1Cat 用它把"短上下文标量图"和"长上下文 XQA 图"分开（`cudagraph_dispatcher.py:60-89`、`:22-24`）。FastLLM 侧对应能力是**图内条件分支**（`fastllm-paged-attention-native.cu:3194-3200` 按 dtype/GQA 选路），不是分图——所以未必有差 | 需先证明"同一 batch 在 8K 与 180K 下需要**不同图**"，而不是同一图内的运行时分路 | 每加一个桶就多一套 `reservedPointers`；`:4346-4348` `graphCount*16MB` 已按档位数线性记费 | **未做、未验证必要性** | 【我的推断】 |
| **L7. 修 `sm70_qwen38nvfp4_1cat_gap_audit.md` 的过期 P0-1** | **不是性能路线，是文档纠错。** 该文档 `:87-96` 说"SM70 graph 被策略关掉（要 cap>7.5）"，而现行代码门是 `>= 70`（`util.py:317`、`:332`），Volta 只留显式 opt-out（`:339-343`） | 已能证实 | 若照旧文档立项会白做工 | **文档待改** | 【代码核实】 |

**关于 L2 的"per-collective 主机同步是否会从根上挡住 prefill 成图"——结论：不会。**
`FastllmNcclPostSyncEnabled()`（`src/devices/multicuda/fastllm-multicuda.cu:2445-2464`）在
`FastllmCudaGetNcclForceSync()` 为真之后**还会**查 `cudaStreamIsCapturing`，捕获态直接返回 false：
```
// 流捕获期间必须返回 false：cudaStreamSynchronize 属于捕获期间被禁止的调用，会直接
// invalidate 正在进行的 CUDA graph 捕获；捕获期间集合通信只是被录制、并未真正执行，
// 不存在与真实 cudaMalloc 争用驱动锁的死锁风险，跳过同步是安全的。
```
同一文件 `:3164-3173` 的 `waitForRanks` 也在捕获态把 `rendezvous` 置空；
`include/devices/multicuda/ncclsubmitrendezvous.h:12-13` 写明
"Capture must bypass BOTH boundaries; replay does not execute this host code at all"。
**所以捕获期与回放期都不执行主机同步；这条门只在 eager 路径上生效。**
L2 的障碍不在 NCCL 门，而在上面的 **chunk 形状可变 + 每 chunk 重新分配 KV 页** 两条结构性原因。

---

## D. 一句话结论

**不建议动这块，理由是"已被实测卡死"而非"没想清楚"：**
FastLLM 的 decode 图（档位、几何预捕获、指针表、跨 rank 表决、捕获后拓扑重写）已经**不落后于**
1Cat，唯一实质缺口是"prefill 不进图 + 没有可打断图"，但这两者的收益在 SM70/4卡 TP4 上
**已被本项目的实测否掉**——设备侧 `max_concurrent=1`、`t_>=2=0.0000 s`
（`docs/sm70_4x200k_rotate_plan.md:383-384`），且把 SM70 的 per-collective 主机同步降 45%
后墙钟纹丝不动（`.audit/sm70-longctx-kv.tsv:85`）；180K prefill 的稳态设备空闲只有 **3.17 s（约 3%）**，
其中 43/44 个 >10 ms 间隙还是 NCCL 内部的 allreduce→broadcast 交接、**不是等主机**
（`docs/sm70_4x200k_rotate_plan.md:443-451`），所以"把 prefill 也烤进图"能捡的钱**上界 <0.5 s**，
而 1Cat 的 breakable 机制换到 FastLLM 要跟现有单事务图池
（`fastllm-cuda.cu:5057-5135`）和跨 rank 表决（`qwen3_5.cpp:3667-3778`）重做，改动面与收益不成比例。

**唯一我建议真做的两件事，都不是"移植 1Cat"：**
（1）**文档纠错**（L7）：`docs/sm70_qwen38nvfp4_1cat_gap_audit.md:87-96` 的 P0-1 把 SM70 图说成被
`cap>7.5` 关掉，现行代码门是 `>= 70`（`tools/fastllm_pytools/util.py:317`、`:332`），照旧文立项会白做工。
（2）**补一条低成本的 correctness/覆盖缺口**（L1）：FastLLM 的解码图按**精确 batch** 建 key
（`qwen3_5.cpp:3970-3973`）且**没有任何向上取整**，而 1Cat 有预计算档位表
（`cudagraph_dispatcher.py:291-310`）。在 MTP 打开时 FastLLM 的档位被砍到只剩 batch=1
（`:4252`），此时任何 batch≥2 的 decode 都整步走 eager——**这笔损失有明确来源但需要 trace 量化**，
建议先量 `batch` 分布再决定是否补 pad-up，不要先改代码。
