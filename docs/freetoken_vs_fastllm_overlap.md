# FreeToken 的重叠机制哪些能搬到 FastLLM，各值多少钱

日期：2026-09-17
对象：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / PCIe（无 NVLink）/ TP4
对照实现：FreeToken `/tmp/FreeToken` @ `cac247a`
本文性质：**只读源码与文档对账 + 读已有 nsys 轨迹**。没有跑任何 GPU 测试，没有改任何源码。

**证据等级**（本文每条数字必须能回答"这是在哪测的"）：

- **[目标系统实测]** 在 FastLLM 上跑出来、且这次由我读 `/home/nsys/*.sqlite` 现算的数。附文件与算法。
- **[引自 FastLLM 文档]** 引自本仓库文档/日志里的数。
- **[我的估算]** 我算的，算法写在原地。
- **unknown** 没有数据。

---

## 0. 结论先行

1. **FreeToken 那三套重叠能成立，靠的不是"两条流"，是"两个硬件引擎"。**
   移动的那一侧跑在**拷贝引擎**（DMA，负责 H2D 和 PCIe 零拷贝 gather）或**主机 CPU 核**上，
   计算那一侧跑在 **SM** 上。两边抢的是带宽，不是 SM 发射槽位，所以能同时干活。

2. **FastLLM 的 R1/R2/R5 之所以负收益，根因是它把 NCCL 集合通信当成了"移动"。**
   集合通信自己就是一个 SM 内核（`ncclDevKernel_AllReduce_Sum_f16_RING_LL`）。
   两条流上放两个 SM 内核 = 两个内核抢同一批 SM，不是重叠。
   **这是"移动"这个词的分界线：能上 DMA 的才叫移动，走 SM 的都是计算。**

3. **FastLLM 上没有 FreeToken 式重叠的可捡空间，而且我有直接测量。**
   在已有轨迹里，稳态 prefill 段 SM 96.93% 忙（180K，91.0 s 窗口），
   而且**拷贝引擎干的那点活已经全部落在这些空隙里、没有跟 SM 抢**：
   拷贝与内核的重叠时间是 **0.000000 s**（180K prefill）；
   拷贝引擎的活只占稳态窗口的 **0.574%**。
   **就算把拷贝引擎的活全部藏起来，能回收的上限也只有 0.6%。**

4. **真正还剩的一条，不是"重叠"，是"把一次 40 MiB 的阻塞式 H2D 变成不阻塞"。**
   180K prefill 里每卡有 **47 次** 41 943 040 B 的 H2D（正好 = 4096 token × 5120 hidden × 2 B），
   是可分页内存的**同步** `cudaMemcpy`。单次设备侧 **7.00–7.53 ms**，
   主机侧阻塞 **7.27 ms**，47 次合计主机阻塞 **341.6 ms**。
   它**独占 PCIe**（等效 5.99 GB/s，只有 PCIe3 x16 实用峰值的一半），
   且与内核重叠 **0.000000 s**。
   这条与 FreeToken 的 `decode_submit/decode_sync` 同构（都是"先发射、别等"）。
   量级：设备侧 0.279 s / 91.0 s = **0.307%**；主机侧 0.3416 s / 118.2 s = **0.289%**。

5. **侧车那条大 D2D 拷贝是我自己一开始搞错的地方，登记在此。**
   我先前把 `dec80k_post.sqlite` 整段当成 decode，于是把 10 MiB/6 MiB 的 2D D2D
   算进 decode 稳态。**这是口径错误。** 用 `CustomAllReduce` 内核密度切段后，
   那张轨迹的 58.1 s 里有 **40 s 是 prefill**（`nvfp4_qpn2_sm70_kernel` 计数为 0，
   AR 全是 5.4 ms 的 41.9 MB 形态），只有约 **2.6 s 是真 decode**。
   那 1848 次 10 MiB + 1848 次 6 MiB 的 D2D **发生在 prefill 段，不是 decode**，
   折算 **每 chunk 各 192 次**。真 decode 段（[56,58.6) s，约 127 token）里
   大 D2D（≥1 MiB）只有 **18 次、合计 0.00047 s**。
   **所以"把侧车搬出 decode"这条候选不成立**，它本来就不在 decode 里。

**一句话：FreeToken 的重叠在 FastLLM 上无处可搬，因为 FastLLM 的瓶颈是 SM 忙，
不是引擎闲着。剩下唯一像 FreeToken 的一条（阻塞 H2D 改异步）收益上限 0.6%。**

---

## 1. 机制对照表（一页）

三个词的第一个字都是**引擎**，别搞混：**拷贝引擎**（copy engine，DMA）是 GPU 上独立于 SM 的搬运硬件，
`cudaMemcpy`/`cudaMemcpyAsync` 的 H2D/D2H/D2D 通常由它执行，不占 SM 发射槽。
**SM**（streaming multiprocessor）是跑计算内核的单元。**join** 指两条并行的工作
重新汇合、必须互相等待的那个同步点。

下表三列是 FreeToken 的三套重叠。源码路径都相对 `/tmp/FreeToken/python/freetoken/`。

| | 重叠 1：整层权重流式（prefill） | 重叠 2：MoE 专家层双缓冲（prefill） | 重叠 3：混合 decode（CPU 溢出专家） |
|---|---|---|---|
| 源码 | `models/weight_stream.py:94-118` | `moe/offload_cache.py:668-816` | `layers/moe.py:298-345` |
| 引擎 A | 拷贝引擎（H2D，`non_blocking=True`） | 拷贝引擎（H2D）+ SM（命中行的 `fast_index_copy` gather） | 拷贝引擎（H2D/D2H）+ **主机 CPU 核** |
| 引擎 B | SM（当前层的 GEMM） | SM（上一层/这一层的 GEMM） | SM（GPU 专家 GEMM） |
| 争的资源 | PCIe 带宽 vs HBM 带宽 | PCIe 带宽 vs HBM 带宽 | PCIe 带宽 vs 主机 DRAM 带宽 |
| join 长什么样 | `compute.wait_event(ready_events[buf])`，就一层（`models/weight_stream.py:111`） | `current_stream.wait_event(prefill_ready_events[buffer_id])`（`moe/offload_cache.py:829`），配上 `release_events` 防止覆写（同文件 `:839`） | 主机侧 `decode_submit` 先发射、`decode_sync` 后收（`moe/cpu_executor.py:543,589`），中间夹 GPU 的 fetch + GEMM |
| 缓冲深度 | 2（`staging = (2, row_bytes)`，`models/weight_stream.py:58`） | 2（借用 slot cache 前 2E 个槽） | 无缓冲，靠"提交在前、等待在后" |
| FastLLM 里对应物 | 有，且已经实现了（按层流式加载，`src/models/qwen3_5.cpp:25351`）；但只在**加载期**，不在前向期（见 §2.9） | 没有。FastLLM 的 MoE 走 cache/MergeMOE，而目标模型是 dense（依据见 §5） | 有，且在跑（NUMA MoE 的 `decodeInputPrefetch`，`src/devices/numas/numasdevice.cpp:1677`），但目标模型是 dense，走不到 |

**为什么重叠 2 的 gather 那一格值得单独看一眼。** FreeToken 的 gather 是
`fast_index_copy` 内核（`kernel/csrc/jit/fast_index_copy.cuh:201,487`），
它读的是**映射过的 pinned 主机内存**，
所以它虽然是个 SM 内核，读的那一头不过 SM，走的是 PCIe。

FastLLM 也有同构的东西：`include/devices/cuda/fastllm-cuda-record-copy.cuh`
（注释写明 Source may be mapped pinned host memory，`:11-13`），从 host 记录表拉专家。
两边是同一个招。

---

## 2. FastLLM 候选清单（movement 形状的活）

"movement 形状"指这件事的**主体是搬字节**，不是算数。下面每一条都给了我在盘上
核到的位置。**核不到的我就不写，并在 §2.9 说明找过什么。**

### 2.1 阻塞式 H2D：chunk 级 41.9 MB 上传（最像 FreeToken 的一条）

- 位置：`src/devices/cuda/fastllm-cuda.cu:5900`（`FastllmCudaCopyFromHostToDevice` 里是 **同步** `cudaMemcpy`）
- 实测：`/home/nsys/pf180k.sqlite`，device 0，41 943 040 B 的 H2D 共 **47 次**，
  单次**设备侧 7.001–7.532 ms**，**主机侧阻塞 7.27 ms（均值）**，由 `cudaMemcpy_ptds_v7000` 发出。
  47 次合计：**设备侧 0.3358 s、主机侧 0.3416 s**。
- 这个字节数是整数账：**4096（chunk）× 5120（hidden）× 2 B = 41 943 040 B**，一分不差。
  所以它是**每个 chunk 一次的激活/输入上传**。
- 等效带宽 **41 943 040 / 7.0 ms ≈ 5.99 GB/s**，
  只有 PCIe3 x16 实用峰值（约 12.5 GB/s）的**一半**。可分页内存的同步拷贝就是这个效率。
- **它与内核重叠 0.000000 s**（180K prefill 段实测），也就是说这 7 ms 里 SM 是空的。

### 2.2 大块 2D D2D 拷贝（prefill 段，每 chunk 各 192 次）。先更正我自己的口径错误

- 位置：调用点是 `src/devices/cuda/linear/fastllm-linear-fp8.cu:3646`
  （`Nvfp4QpnPrepareFromNative`）；打包内核在 `src/devices/cuda/sm70/qpn2_nvfp4.cu:95,483,486`
- **我一开始把 `dec80k_post.sqlite` 整段当成 decode，这是错的，登记在此。**
  用 `CustomAllReduce` 内核（decode 独有）的密度切段后，那张轨迹 58.1 s 里
  **t=12..56 s 是纯 prefill**（`nvfp4_qpn2_sm70_kernel` 计数为 0，
  AR 内核全部是 5.4 ms 的 41.9 MB 形态），**只有约 2.6 s 是真 decode**。
- 更正后的实测：
  - **prefill 段**（`dec80k_post.sqlite` [14,54) s）：10 MiB D2D **1848 次**、
    6 MiB D2D **1848 次**，合计设备时间 **0.0956 s / 40.0 s = 0.239%**。
    按 `Qwen35QGateKVPrefill` 内核 616 次 / 64 层 ≈ 9.6 个 chunk 折算，
    **每 chunk 各 192 次**。发出方是 `cudaMemcpy2D_ptds_v7000`（同步 2D）。
  - **真 decode 段**（同轨迹 [56.3,58.5) s）：大 D2D（≥1 MiB）**0 次**。
- 所以"把侧车从 decode 挪走"这条候选**不成立**，它本来就不在 decode 里。
  这 192 次/chunk 的拷贝是 prefill 期按 projection 摊出来的搬运。

### 2.3 权重加载期 H2D

- 位置：`src/models/qwen3_5.cpp:7727`、`src/model.cpp:5512`（组间串行、组内并行）
- 实测：`/home/nsys/dec80k_post.sqlite` 里 2.543 GB 的单次 H2D，四卡各一次，
  每卡 **0.367–0.390 s**。**注意这是启动期的 embedding 表上传，不是稳态**，
  且在 kernel 窗口起点之后（traces 的 t≈3.1/4.6/6.0/7.3 s），
  所以它落在 warmup 区、**不能算稳态开销**。
- 该路径已在 `docs/qwen35_streaming_load.md` 记为已落地（`malloc_trim(0)` 在
  `qwen3_5.cpp:25548`）。

### 2.4 权重重排 / 布局转换（NVFP4 native ↔ QPN2 侧车）

- 位置：`fastllm-linear-fp8.cu:3616`（`FastllmCudaEnsureNVFP4Qpn2Layout`）、
  `:3636`（`Nvfp4QpnPackedBytes`）、`qwen3_5.cpp:9700`（warmup 时一次性建全部侧车）
- 字节账：native 4.252 GiB/rank + 侧车 3.194 GiB/rank
  **[引自 FastLLM 文档]** `docs/sm70_load_path_fastllm_vs_1cat.md:109`
- 关键事实，**它只在 warmup 发生一次**：`nvfp4Qpn2Wanted` 是粘性标记
  （`fastllm-linear-fp8.cu:3595`），一旦立起来就不会重排。
  **所以前向期没有重复 repack。**

### 2.5 KV 页分配、淘汰、压缩

- 位置：`include/fastllm.h:713`（`PagedCacheManager`）、`GetUnusedPageIndex` 在
  `multicudadevice.cpp:1238`、`ReleasePageIndices` 在 `:1253`、`:1399`；
  页池上限 `qwen3_5.cpp:22187`（`pagesLimit = totalPages * 4 / 5`）
- 页内数据的搬运用的是 `FastllmCudaMemcpy2DDeviceToDeviceAuto`
  （`multicudadevice.cpp:1201`、`:1304`、`:1389`，最终落到
  `fastllm-multicuda.cu:89` 的同步 `cudaMemcpy2D`）
- **一个要登记的疑点**：`CopyPagedCacheSliceToDense`（`multicudadevice.cpp:1266`）和
  `EnsureDenseMirrorFromPagedCache`（`:1312`）**全仓只有定义、没有调用点**
  （grep 全树只命中这两处 + 内部互调）。所以"页→稠密镜像的拷贝"在当前树里是死代码，
  不能算候选。

### 2.6 跨卡 P2P 拷贝

- 位置：`fastllm-cuda.cu:6131`（`FastllmCudaMemcpyBetweenDevices`）、
  `:6240`（`FastllmCudaMemcpyPeerAsyncCurrentThread`）
- 实测：`/home/nsys/dec80k_post.sqlite` device 0 的 `CUDA_MEMCPY_KIND_PTOP` 共 381 次，
  **全是 4 字节**，总设备时间 0.0011 s。这是 token handoff 的信号量，不是数据搬运。
- 另有 `cudaMemcpyPeerAsync_v4000` 381 次、4 字节（同上）。
- **结论：这条在目标配置上量级为零，不是候选。**

### 2.7 embedding 主机侧查表 + H2D

- 位置：`qwen3_5.cpp:15276`（`useCpuEmbedding`）、`:28118`（draft 侧同样的判断）
- **[引自 FastLLM 文档]** `.audit/sm70-longctx-kv.tsv` 第 46/50 行：
  把表挪到主机省 2.4–2.55 GB/卡，速度不降反升 1–3%，TTFT 变差 0.8–1.3%
- 状态：**已拿**（`--low_gpu_mem`）。

### 2.8 NUMA MoE 的 D2H 预取（FastLLM 自带的、和 FreeToken 同构的重叠）

- 位置：`src/devices/numas/numasdevice.cpp:1677`（`decodeInputPrefetch` 的发射）、
  `:1694`（`FastllmCudaEventRecordCurrentThread(sourceReadyEvent)`）、
  `:4793`（结果回写的 `FastllmCudaCopyFromPinnedHostToDeviceAsync`）
- 机制：另起 `inputCopyStream`/`routeCopyStream`（`:1566,:1585`），把激活和路由异步 D2H，
  让 CPU 专家算的同时 GPU 继续。
- **和 FreeToken 重叠 3 是同一招**：分开"提交"和"等待"。
- 目标模型是 dense，**这条路径到不了**。

### 2.9 我找过但没找到的东西

- **前向期的整层权重重流（Forward-time weight streaming）**：搜了
  `streamingCudaLoadEnabled`、`ShouldLoadWeightSeriallyBeforeOthers`、
  `PrepareStreaming*Layer`，全部命中加载期路径（`qwen3_5.cpp:24866-25548`），
  没有前向期调用点。**[引自 FastLLM 文档]** `.audit/sm70-longctx-kv.tsv` 第 52 行
  已经算过：每步流 4.2 GB / 约 12 GB/s ≈ 350 ms/token，对当前约 11 ms/token 是自杀。
- **KV 页的批量压缩/搬运 kernel**：搜 `Compact`、`compact` 全树，
  跟 KV **数据搬移**有关的只有 `FastllmCudaDFlashCompactKVCache`（`fastllm-cuda.cu:15338`），
  那是 **DFlash draft** 的滑动窗口裁剪（调用点 `qwen3_5.cpp:27812`），
  而且是 gather→scratch→scatter 两个 **SM kernel**（`:15398,:15403`）。
- **KV 页写入本身也是 SM kernel，不是拷贝引擎**：`FastllmPagedCacheCopyKernel`
  （`attention/fastllm-attention.cu:2503`，启动点 `:2567`）与
  `FastllmPagedCacheCopyMultiPageKernel`（同文件 `:2609`，启动点 `:2673`），
  调用者是 `cudadevice.cpp:10488,10511,10539`、`multicudadevice.cpp:1173,1184`。
  两者启动后都直接 `DeviceSync()`（`:2572`、`:2677`）。
  **这条很重要**：它说明 KV 页搬运走的是 SM，**不是**可重叠的拷贝引擎。
  **想靠侧流藏它，会撞上 §5 的同一条墙。**
- **KV 页池的 CPU 侧（kvCacheInCPU）**：被硬断言挡住
  （`qwen3_5.cpp:15285` 的 `AssertInFastLLM(!GetKVCacheInCPU(), ...)`），
  目标配置走不到。

---

## 3. 收益表

单位与工况都写在行内。**"已拿"是已经落地进基线的，"不做"是结论否掉的。**

| # | 候选 | 机制 | 预期收益（数字 + 怎么来的） | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|---|---|
| **C1** | 大块 2D D2D（prefill，每 chunk 各 192 次）改异步 | 现在走 `cudaMemcpy2D_ptds_v7000`（同步 2D，`fastllm-multicuda.cu:89`），改成异步版本 | 上界 = 这部分拷贝的设备时间：**0.0956 s / 40.0 s = 0.239%**（80K prefill 段）**[目标系统实测]**，`/home/nsys/dec80k_post.sqlite` device 0 [14,54) s，`copyKind=8 AND bytes>=1048576` 的 `SUM(end-start)` | 异步 2D 版本已存在（`fastllm-cuda.cu:5963` `FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread`），机制可达 | 收益 0.24%，低于噪声底；且这些拷贝已在空隙里、不占 SM | **不做**（收益小于测量分辨率） |
| **C2** | chunk 级 41.9 MB H2D 改异步 | FreeToken 的 `decode_submit/decode_sync` 同构：先发射、后等 | 设备侧上界 **0.3358 s / 91.0 s = 0.307%**（180K prefill）；主机侧阻塞 **0.3416 s / 118.2 s = 0.289%** **[目标系统实测]**，`/home/nsys/pf180k.sqlite` device 0，41 943 040 B 的 H2D 共 47 次、单次设备 7.00–7.53 ms、主机 7.27 ms | 该拷贝确实阻塞主机（`cudaMemcpy` 同步版，`fastllm-cuda.cu:5900`），实测与内核重叠 0.000000 s；改 `cudaMemcpyAsync` 到独立流即可 | 设备稳态 96.93% 忙（180K，91.0 s 窗口实测）；参照 `.audit:85` 的同类改动：同步腰斩、墙钟不动。**主机不再阻塞 ≠ 买得到吞吐** | **不做**（同 C3 的教训） |
| **C3** | 通用"把阻塞拷贝挪到侧流" | 抬掉 `NcclForceSync` 那类主机同步 | **实测量级为零**：`.audit` 第 85 行记 `FASTLLM_SM70_NCCL_ASYNC=1` 同步 −45%、`max_concurrent` 仍 1、墙钟 0 **[引自 FastLLM 文档]**；我这次复算设备侧 `max_concurrent=1`（180K prefill）、拷贝与内核重叠 0.0000 s **[目标系统实测]** | 需要设备真有 >2% 的空闲 | 已经做过、已经否掉 | **已否** |
| **C4** | 权重加载期读盘与 H2D 重叠 | 读 I/O 与 H2D 并发 | **unknown**。加载耗时不是瓶颈：进程起→可服务约 **41.9 s** **[引自 FastLLM 文档]** `docs/sm70_load_path_fastllm_vs_1cat.md:124` | 需要加载时长成为矛盾 | 该文档 §D 已判"加载侧没有可捡的收益" | **已否** |
| **C5** | 跨卡 P2P 数据搬运 | 用 `cudaMemcpyPeerAsync` | **量级为零**：PTOP 381 次全是 4 字节，设备时间 0.0011 s **[目标系统实测]** | 不适用 | 不适用 | **不做** |
| **C6** | KV 页写入走 SM，不是拷贝引擎 | 页写入是 `FastllmPagedCacheCopy*Kernel` + `DeviceSync()` | **不是"收益小"，是"做了没用"**。它是 SM kernel（`attention/fastllm-attention.cu:2503,2609`），挪到侧流等于重做 R1/R2，撞 §5 那条墙。而且量级本来就近零：prefill 段 device 0 上 `PagedCacheCopy*` 1492 次 / **10.247 ms / 0.026%**，`PagedCacheCopyMultiPage` 1232 次 / **9.773 ms / 0.024%** **[目标系统实测]**，`/home/nsys/dec80k_post.sqlite` [14,54) s。可重叠的页→稠密镜像路径（`multicudadevice.cpp:1266,1312`）**无调用点** | 需要先把页写入改成拷贝引擎的活 | 方向本身就错 | **已否**（结构性，且量级 0.03%） |
| **C7** | 拿掉 native NVFP4 那一份布局 | 省显存，不省时间 | 省 **4.566 GB = 4.252 GiB/rank（卡的 26.6%）** **[引自 FastLLM 文档]** `sm70_load_path_fastllm_vs_1cat.md:109` | 先搬 1Cat 的 QPN2-packed prefill 分派 | 是显存账不是时间账；QPN4 已被 `sm70_status_and_backlog.md:128,469` 判"不进默认范围" | **未做**（换显存，非重叠） |

**收益表的一句话读法。** C1 的 0.239% 与 C2 的 0.307% 都是**上界**，而且它们的前提
是"设备本来有空"，而实测设备稳态 **96.93% 忙**（180K，91.0 s 窗口，§4）。
所以这两条的**真实**收益必然更小，大概率贴零。C3 是已经做完并实测为 0 的同类，
它给出的教训直接适用：**主机不再阻塞 ≠ 买得到吞吐。**

**这张表里没有一行是可做的。** 原因不是我不愿意找，而是三条都撞同一堵墙：
FastLLM 的稳态 prefill 段设备已经 **96.93% 忙**，
**拷贝引擎的活加起来只占该窗口的 0.574%（0.5220 s / 91.00 s），
而这 0.5220 s 已经全部落在 SM 的空隙里**（与内核重叠 0.000000 s，§4）。
把拷贝全藏起来，上界就是这 0.574%；而 180K prefill 的空隙总量只有
2.791 s（3.07%），其中拷贝占掉 0.5220 s（**18.7% of gap**）。
**没有可捡的空间。**

---

## 4. 这次读轨迹的算法（可复现）

全部只读 `/home/nsys/*.sqlite`，用 Python 的 `sqlite3` 模块（本机没有 `sqlite3` 命令行）。

```python
import sqlite3
c = sqlite3.connect('file:/home/nsys/pf180k.sqlite?mode=ro', uri=True)
# 1) 窗口
lo, hi = next(c.execute("SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"))
# 2) 切段：不能只用 t>=12.4 s 当"稳态"，因为一张轨迹里 prefill 和 decode 会混在一起。
#    判据：CustomAllReduce 内核（decode 独有）的密度；或 nvfp4_qpn2_sm70_kernel（decode NVFP4）
#    与 AR 内核时长（5.4 ms = 41.9 MB prefill 形态，<50 us = decode 形态）。
# 3) 每设备内核忙时与拷贝忙时
c.execute("SELECT SUM(end-start)/1e9,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? AND start>=? AND start<?", (0,t0,t1))
c.execute("SELECT SUM(end-start)/1e9,COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=? AND start>=? AND start<?", (0,t0,t1))
# 4) 拷贝与内核的重叠：先把内核区间合并成不相交区间，再用 bisect 判定
#    （直接二重循环在 98 万内核 x 3.4 万拷贝上会超时）
# 5) 并发度：对内核区间做扫描线，取峰值 = max_concurrent，累加 cur>=2 的时间 = t_>=2
# 6) 拷贝的发起方：CUPTI_ACTIVITY_KIND_MEMCPY.correlationId 关联
#    CUPTI_ACTIVITY_KIND_RUNTIME.correlationId，取 StringIds 拿 API 名
```

**关键结果（device 0，按段切开）**

| 轨迹 | 段 | 窗口 | SM 忙 | SM busy | max_concurrent | t_>=2 | 拷贝引擎忙 | 拷贝占窗口 | 拷贝与内核重叠 | 空隙 | 空隙被拷贝占掉 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pf180k.sqlite` | 180K prefill | 91.00 s | 96.93% | 88.21 s | 1 | 0.0000 s | 0.5220 s | **0.574%** | 0.000000 s | 2.791 s | 18.7% |
| `dec80k_post.sqlite` | 80K prefill | 40.00 s | 95.79% | 38.31 s | 1 | 0.0000 s | 0.1285 s | **0.321%** | 0.000000 s | 1.685 s | 7.6% |
| `dec8k.sqlite` | 8K prefill | 9.00 s | 89.06% | 8.02 s | 2 | 0.0001 s | 0.0414 s | 0.460% | 0.000013 s | 0.985 s | 4.2% |
| `dec80k_post.sqlite` | 80K C=1 decode | 2.20 s | 77.25% | 1.70 s | 3 | 0.0013 s | 0.0358 s | 1.629% | 0.000361 s | 0.500 s | 7.1% |
| `dec8k.sqlite` | 8K C=1 decode | 2.00 s | 93.61% | 1.87 s | 3 | 0.0011 s | 0.0367 s | 1.837% | 0.000157 s | 0.128 s | 28.6% |
| `c2b.sqlite` | 8K C=2 decode | 2.00 s | 95.16% | 1.90 s | 2 | 0.0002 s | 0.0183 s | 0.913% | 0.000051 s | 0.097 s | 18.8% |

**[目标系统实测]**，来源 `/home/nsys/*.sqlite`，算法如上一段。

**这张表怎么读。** 长上下文的 prefill 段是判断"要不要做重叠"的正确地方：
SM 96.93% 忙、拷贝只占 0.574%、且**拷贝与内核重叠 0.000000 s**（一次都没碰上）。
decode 段的拷贝占比看着高一点（1.6%），但那是**分母变小**造成的：
decode 窗口只有 2.2 s，绝对量是 0.0358 s，和 prefill 的 0.1285 s 比还是更小。

**并发度只有 1（180K prefill，91 s 窗口）**，`t_>=2 = 0.0000 s`。
这直接印证了 §5 那条：**两条流上放两个 SM 内核买不到重叠。**
decode 段 `max_concurrent=3`，但 `t_>=2` 也只有 1.3 ms，是内核交接的毛刺。

**并发度**（同窗口，同类扫描线）：180K prefill `max_concurrent=1`、`t_>=2 = 0.0000 s`；
80K decode `max_concurrent=3`、`t_>=2 = 0.0013 s`（0.003%）；8K decode 同量级。
**也就是说：设备上同一时刻基本只有一个内核在飞，两条流放两个 SM 内核确实买不到东西。**

---

## 5. 明确**不能**搬的东西，以及结构性理由

| 不能搬的东西 | 结构性理由 |
|---|---|
| **FreeToken 式"整层权重在前向期双缓冲流过"** | FastLLM 的权重在 HBM 里常驻，`docs/qwen35_streaming_load.md` 的流式只在**加载期**省主机峰值。前向期每步流一遍的账已经被算过：4.2 GB / 约 12 GB/s ≈ 350 ms/token，对当前约 11 ms/token **[引自 FastLLM 文档]** `.audit/sm70-longctx-kv.tsv:52`。**两边的前置条件不同**：FreeToken 是边缘设备装不下权重，FastLLM 是权重装得下、缺的是带宽重叠空间。 |
| **用侧流藏住 TP all-reduce** | 集合通信是 SM 内核（`ncclDevKernel_AllReduce_Sum_f16_RING_LL`，`pf180k.sqlite` 整段 device 0 上 8588 次、36.140 s，占内核时间 108.453 s 的 **33.3%**；只取 prefill 段 [14,105) 是 5010 次 / 29.454 s / 88.209 s = **33.4%** **[目标系统实测]**）。**两个 SM 内核在两条流上不重叠，只是排队**，而且是排队加事件开销。R2 实测 **−7.04%**，其中 4.37 点来自去掉 host drain、2.37 点是换流的固定开销 **[引自 FastLLM 文档]** `docs/sm70_4x200k_rotate_plan.md:421`。 |
| **FreeToken 式 hybrid decode（CPU 算溢出专家 + GPU 算其余）** | 上一层前提是"有专家要算"。目标模型 **dense**：checkpoint 里 `expert` 张量 **0 个**，config 里**没有** `num_experts`/`moe_intermediate_size`（我在 `/home/models/Qwen3.8-27B-QUASAR-NVFP4/config.json` 与 `model.safetensors.index.json` 上核过）。而且 FastLLM 里唯一那条 CPU 专家路径要求 NUMA：`FastllmCudaMergeMOEHybrid` 的整个函数体在 `#ifdef USE_NUMAS` 里（`src/devices/cuda/moe/fastllm-moe-cache.cu:1544,1547,1661`），并且权重必须已按 NUMA 节点切成 `numasData` + pinned（`BindNumaWeights`，同文件 `:268`，形状门 `:295-296`）。 |
| **FreeToken 式的 `fast_index_copy` 零拷贝 gather** | 前提是"有东西在主机 pinned 内存里、且 GPU 要按索引抓它"。目标配置下权重的每卡 4.18 GiB 全在 HBM **[引自 FastLLM 文档]** `.audit/sm70-longctx-kv.tsv:52`，没有这样的主机侧实体。**但 FastLLM 已经自己长出了同一个招**：`fastllm-cuda-record-copy.cuh:11-13` 明写 source 可以是 mapped pinned host memory。**所以这不是"要搬"，是"两边都有"。** |
| **KV 页写入 / eviction 的重叠** | 两层理由。① **它本来就不是拷贝引擎的活**：KV 页写入走 `FastllmPagedCacheCopyKernel` / `FastllmPagedCacheCopyMultiPageKernel`（`attention/fastllm-attention.cu:2503,2609`），是 SM kernel，而且启动后直接 `DeviceSync()`（`:2572,2677`）。**挪到侧流就是又做一次 R1/R2，撞同一条墙。** ② 量级也小：`CUDA_MEMCPY_KIND_PTOP` 381 次全是 4 字节（实测）；页→稠密镜像的拷贝路径（`multicudadevice.cpp:1266,1312`）**全树无调用点，是死代码**。 |
| **`--low_gpu_mem` 那条（embedding 上主机）当成"重叠"** | 它省的是显存，不是时间。实测 decode 不降反升 1–3%、TTFT 差 0.8–1.3% **[引自 FastLLM 文档]** `.audit/sm70-longctx-kv.tsv:46`。**已拿，但别把它算成重叠收益。** |

---

## 6. 测量计划（针对 C1 与 C2）

C1 与 C2 的收益都在 0.3% 以下，**低于本项目任何一次 A/B 的噪声底**
（`.audit` 第 83 行记录：同一开关四次交替，run-to-run 离散 0.437 s，
效果只有 0.078 s，即 **0.18 倍噪声**）。所以这两条**不能**用端到端墙钟判定，
只能用**设备侧拷贝时间的直接测量**判定。下面给的就是这个口径。

### 6.1 C1：prefill 每 chunk 192 次的大块 2D D2D 改异步

- **本条的前提已经变了**。我最初以为这些拷贝在 decode 里，所以想"把它移出 decode"。
  更正后（§2.2）它是 **prefill 段**、每 chunk 各 192 次的现象，
  所以正确的做法不是"移走"，而是"改成异步 2D"。
- **先做代码核实（零成本，必须第一步）**：这 1848+1848 次拷贝的**调用者**是谁，
  本轮只证到"由 `cudaMemcpy2D_ptds_v7000` 发出、紧跟在
  一个 `cutlass::Kernel2<...h884gemm...>` 之后"，**没有**证到具体源码行。
  要查的调用点包括 `src/models/qwen3_5.cpp:3330,3403,7334,27581`
  与 `src/devices/multicuda/multicudadevice.cpp:1201,1304,1389,1638,2659,2665,3940,4005`。
  **这一步做完之前不许开工。**
- **配置**：`--tp 4 --dtype auto --tokens 200000 --max_batch 1 --gpu_mem_ratio 0.98
  --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size 4096
  --input_tokens 80000 --output_tokens 8 --batch 1 --warmup 0`
  （与 `docs/sm70_4x200k_rotate_plan.md` §8.6 同口径，便于和已有 `dec80k_post.sqlite` 对照）
- **控制**：同一二进制、同一 sha256 的 token 流；对照组就是改前的同一条命令。
- **判定口径（不看墙钟）**：
  1. 跑 nsys，导出 sqlite；
  2. 用 §4 的脚本统计 device 0 **prefill 段**（不是整段）
     `copyKind=8 AND bytes>=1048576` 的 `SUM(end-start)` 与调用数；
  3. **通过门**：该 bucket 设备时间从 **0.0956 s** 降到 **< 0.01 s**；
  4. **兜底门**：token sha256 逐位一致（改了调度不能改数值）；
  5. **收益门**：墙钟不回退（>0.5% 变慢就撤）。**注意这里只设"不回退"门，
     不设"必须变快"门，因为 0.239% 的收益在噪声里根本看不见。**

### 6.2 C2：chunk 级 41.9 MB H2D 改异步

- **先决条件（本轮已用 3 个数字证明可达）**：该拷贝是 `cudaMemcpy` 同步版
  （`fastllm-cuda.cu:5900`），实测单次**设备侧 7.00–7.53 ms、主机侧 7.27 ms**、
  47 次合计**设备 0.3358 s / 主机 0.3416 s**。
  改成 `cudaMemcpyAsync` + 独立流在机制上直接可达。
- **配置**：与 §6.1 同（这条要在 180K prefill 上测，用 `--input_tokens 180000`）
- **控制**：`FASTLLM_CUDA_SYNC` 不设（默认 `false`，`src/fastllm.cpp:355`），
  两边都保持"不逐 op 同步"，隔离出这一次拷贝的改动。
- **量什么**：`/home/nsys/*.sqlite` 里
  `copyKind=1 AND bytes=41943040 AND deviceId=0` 的**主机侧**
  `CUPTI_ACTIVITY_KIND_RUNTIME` 时长与**设备侧** `SUM(end-start)`。
- **通过门**：
  1. 主机侧该 API 的总时长从 **341.6 ms** 降到 **< 5 ms**（证明主机真不再被卡住）；
  2. **该拷贝的设备区间与相邻内核重叠 > 0**（现在重叠是 0.000000 s，§4 表）；
  3. **总墙钟不得回退**。若墙钟回退 >0.5%，按 `.audit` 第 85 行的教训记为
     "主机不再阻塞 ≠ 买到吞吐"，撤掉。
- **反例预警**：`.audit` 第 85 行测过同类改动，结果是
  同步次数 **23 120 → 12 800（−45%）**、总阻塞 **152.87 s → 82.06 s（−46%）**，
  **但 `max_concurrent` 仍 1、`t_>=2` 仍 0.0000 s、墙钟 38.1410 s → 38.2326 s 不动**。
  这条反例直接预示 C2 大概率也是"省主机、不省墙钟"。
  我这次实测的 `max_concurrent=1`（180K prefill，91 s 窗口）与它一致。

### 6.3 我建议实际怎么走

**不要把 C1/C2 立项。** 收益表里两条的上界分别 0.239% 与 0.307%，
都低于 `.audit` 记的 0.437 s 噪声底；而两块地的**先决条件调查成本**
（C1 要先做 §6.1 那条代码核实，C2 要做完整 A/B）都远高于收益。

**真正该做的是把 §4 那张表变成常设口径。** 现在项目里
`max_concurrent` / `t_>=2` 只在 `.audit` 第 65/85 行零散出现过，
而且是从外部轨迹人工读的。§4 的脚本是可复跑的、把三张表
（SM 忙 / 拷贝引擎忙 / 两者重叠）一次打全。

**理由**：判断"某条重叠值不值得做"只需要看那两个数：**0.574%**（拷贝引擎占稳态 prefill 窗口）
与 **96.93%**（SM 占同一个窗口）。
把它固化下来，以后任何"用侧流重叠一下"的提案，跑一次就有数，
不用再重复 R1/R2/R5 那三轮投入。

---

## 7. 撤错与不确定项

**撤错留痕（本文自己犯的错，必须写明原数错在哪）**

- **撤错 1：把 10 MiB/6 MiB 的 2D D2D 当成 decode 现象。**
  - **原数**：写入本文初稿的 C1 行是"decode 稳态里每 token 1.21 次大 D2D，
    设备时间 0.0616 s / 45.70 s = 0.13%"。
  - **错在哪**：我把 `dec80k_post.sqlite` 的**整段 58.1 s** 当成 decode，
    实际那张轨迹里 **t=12..56 s 是纯 prefill**（该段 `nvfp4_qpn2_sm70_kernel` 计数为 0，
    AR 内核全部是 5.4 ms 的 41.9 MB 形态），真 decode 只有约 2.6 s。
  - **来自哪**：我自己对轨迹的分段判断，来自一个错误假设
    （"文件名叫 dec80k 就整段是 decode"），没有做相位核对。
  - **更正后**：这 1848+1848 次拷贝在 **prefill 段**，每 chunk 各 192 次；
    真 decode 段的大 D2D（≥1 MiB）**0 次**。
  - **下游影响**："把侧车搬出 decode"这条候选**整条作废**（§2.2、§3 的 C1 已改写）。
- **撤错 2：用 `t>=12.4 s` 一刀切当"稳态"。**
  - **原数**：初稿 §4 表里的 "45.70 s / 95.8% / 0.39% / 9.1%" 三行。
  - **错在哪**：一张轨迹里 prefill 与 decode 是分开的段，
    用单一时间点切会把两种形状混成一个数。**口径错，不是算错。**
  - **更正后**：§4 表改成按段列（用 `CustomAllReduce` 密度判 decode），
    6 行分别给出 SM 忙、`max_concurrent`、`t_>=2`、拷贝占比、重叠量。
- **两处更正的副产品**：`max_concurrent` 与 `t_>=2` 现在有了分段实测值，
  其中 **180K prefill 段 `max_concurrent=1`、`t_>=2 = 0.0000 s`（91 s 窗口）**，
  这是本文"两个 SM 内核在同一设备上不会重叠"的最直接证据。

**不确定项**

- **不确定项 1**：那 1848+1848 次大 D2D 拷贝，我**没能定位到源码行**。
  只能证到"由 `cudaMemcpy2D_ptds_v7000` 发出、紧跟在
  一个 `cutlass::Kernel2<...h884gemm...>` 之后"。
  指向 QPN2 侧车打包的依据是：尺寸（10 MiB = 一个 projection 级别的块）、
  以及 `/home/nsys/dec80k_post.sqlite` 里 stream 46 上有
  `nvfp4_qpn2_prepack_codes_from_native_kernel` **256 次**。
  **这是推断，不是证据。** §6.1 把它列为开工前的必做核实。
- **不确定项 2**：分段边界是我按 `CustomAllReduce` 密度人工切的（整数秒），
  切点附近可能把几帧混进相邻段。影响很小（拷贝占比都在 0.3–1.8% 量级），
  但**不是严格的分段**。
- **不确定项 3**：8K **prefill** 段 SM 只有 89.06% 忙、8K **decode** 段 93.61%，
  都比 180K prefill 的 96.93% 低。也就是说**短上下文还有 3–11 个点的空**。
  这与 180K 是两种形状。本文"没有重叠空间"的结论
  **对 180K prefill 最硬**，对短上下文弱一些。**短上下文那一档本轮没测，没有结论。**
- **不确定项 4**：`.audit` 第 85 行的 R3 是"别人的数"，不是我量的；
  我只独立复算了同一结论的设备侧部分（`max_concurrent=1`）。
