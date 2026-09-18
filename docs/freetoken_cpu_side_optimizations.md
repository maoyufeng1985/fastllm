# FreeToken 源码里的 CPU 侧优化机制（及在本机的适用性）

**目的**：FastLLM 的吐字阶段已实测为**主机侧受限**（每 kernel 主机成本 20.3 µs > 显卡 12.15 µs，
940 个 kernel/token，逐步发射路径下显卡空闲 42.6%——见
`docs/sm70_decode_prefill_bottleneck_plan.md` §10）。本文从 FreeToken 源码里找 CPU 侧
可借鉴的机制，**逐条核实在本机的适用性**。

**源码**：`/tmp/FreeToken`（Python 561 个文件、125163 行）。

---

## 结论先行

| # | 机制 | FreeToken 出处 | 本机适用性 | 对应 FastLLM 的实测问题 |
|---|---|---|---|---|
| 1 | **融合小拷贝**（预计算描述符，一次发射代替 N 次） | `moe/offload_cache.py:367-443` | **适用** | 吐字窗口 **2352 次 `cudaMemcpy2D`**，占显卡空闲 **12.0/59.2 ms** |
| 2 | **主机内存驻留**（pin-after-fill / mlock） | `moe/host_banks.py:130-175` | **适用** | `--low_gpu_mem` 的流式权重路径**没有 pin** |
| 3 | **异步提交后重叠**（submit → 排 GPU 活 → sync） | `moe/cpu_executor.py:543+` | **部分适用** | 引擎每个集合通信后有强制排空（SM70 死锁防线，不能简单去掉） |
| 4 | **线程绑到物理核、避开 SMT 兄弟** | `moe/cpu_executor.py:92-143` | **❌ 不适用** | **本机 SMT 已关**：cpu0–9 的 `thread_siblings_list` 都只有自己 |

---

## 1. 融合小拷贝（最直接可用）

**FreeToken 做法**（`moe/offload_cache.py:380-443` `_build_fused_copy_plan`）：
把"每个 bank 一次拷贝"换成**一次融合的多 bank 拷贝**，做法是先算好描述符：

- 每行的字节数（`feat`）与各 bank 的基址**只算一次**，缓存的整个生命周期内地址不变
  ——源码注释原话："the addresses are fixed for the cache's lifetime so the descriptor
  tensors are CUDA-graph safe"；
- 逐个检查 `feat % 16 == 0` 与基址 `% 16 == 0`，**不满足就退回逐 bank 拷贝**
  （`_copy_fused_ok = False`）；
- 有一条硬约束：`_build_copy_plan` 里若某 bank 的行字节数不是 128 的倍数，
  **直接报错**，因为在融合被禁用时无法搬运它。

**本机对应的问题（实测）**：吐字窗口内 `cudaMemcpy2D` **2352 次、共约 3 MB**
（24 B × 672、3072 B × 672、1024 B × 672、5120 B × 336），CPU 侧合计 **18.45 ms**，
其中 **12.04 ms 落在显卡空隙里**、平均每次 **7.8 µs**（比发射一个 kernel 的 5.5 µs 还贵）。
**3 MB 数据搬 139 ms（21 MB/s）——瓶颈是调用次数，不是带宽。**

**可借鉴的点**：这些拷贝的源/目标地址在 decode 期间是固定的（同一批 KV/状态缓冲），
符合 FreeToken 那条"描述符只算一次"的前提；把 2352 次调用压成每步一次融合拷贝，
按 §10 的账，可动的部分是那 12.0 ms（**占显卡空闲 20.3%、占吐字窗口 8.6%**）。

## 2. 主机内存驻留（pin-after-fill / mlock）

**FreeToken 做法**（`moe/host_banks.py:130-175`）：

- `pin()`：**填完数据之后**再 `cudaHostRegister`（注释写明 "registered pages cannot be
  dropped"，所以不能提前 pin）；`FREETOKEN_SKIP_BANK_PIN=1` 只给 CPU-only 工具用；
- `lock()`：`mlock` 让页面常驻，**不占 CUDA pin 配额、但没有设备地址**——只有 CPU 执行器
  能服务这种"locked"层；
- `release()`：`madvise(MADV_DONTNEED)` 丢页面但保留地址空间；
- **失败模式写得很清楚**：`mlock` 撞 `RLIMIT_MEMLOCK` 时**警告一次并留作 pageable**，
  后续所有消费者按 pageable 处理（`_os_lock_failed` 是粘性标志，避免刷屏）。

**本机对应**：FastLLM 有 `FastllmCudaHostMalloc`（`cudaHostAlloc`，`fastllm-cuda.cu:6059`）
与 `FastllmCudaHostRegister`（`:6073`），但**调用点只在两处**：
`llmsamplingblock.cpp:115/116/153`（采样的 host ids/scores）与
`numas` 设备（`numas.cpp:215`、`numasdevice.cpp:1475/1487/3864/3873`）。
**`--low_gpu_mem` 的流式权重路径没有 pin。**

**可借鉴的点**：把流式权重占的主机内存按 pin-after-fill 处理。**但要先量**
pageable 与 pinned 的差别（本会话已量过链路层：H2D pinned 12.4 GB/s vs pageable 11.17 GB/s，
引自本会话 `/tmp/h2d_vs` 一类探针——**那是链路带宽，不是"主机侧读权重的延迟"**，
不能直接外推）。

## 3. 异步提交后重叠（submit → 排活 → sync）

**FreeToken 做法**（`moe/cpu_executor.py:543+`）：

`decode_submit()` 只做三件事就返回，不等结果：
1. 把这一步的激活/路由用 `non_blocking=True` 拷进 **pinned** 主机内存；
2. 提交 CPU 线程池任务；
3. 立刻返回一个 handle。

源码注释把意图写得很明确："Lets a caller (the hybrid backend) enqueue GPU work between
this and `decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch"，
并且"输出张量在这里分配，好让它在整个重叠窗口内保持存活"。
另有 `_gpu_prequant` 分支：先在 GPU 上做量化往返，**让 CPU 侧读到的已是量化后的激活，
从而跳过它自己那段串行的标量处理**。

**本机对应与限制**：FastLLM 恰恰相反——SM70 上每个集合通信后强制
`cudaStreamSynchronize`，那是**防跨 rank 死锁的载重设计**（引擎里那条注释写明：
无 P2P 的多卡下，集合通信在途时另一线程触发真实 `cudaMalloc` 会与 NCCL proxy 争驱动锁）。
本会话实测过去掉排空的效果（R2：`37.3322 → 39.9610`，**反而慢 7.04%**，
引自 `docs/sm70_4x200k_rotate_plan.md:437-441`）。**所以这条只能取其"先量化再交给 CPU、
减少 CPU 串行段"的思路，不能照搬"不等待"。**

## 4. 线程亲和性——本机不适用（已核实）

**FreeToken 做法**（`moe/cpu_executor.py:92-143`）：读
`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`，**每个物理核只取一个逻辑 CPU**
并绑上去。理由（源码原话）："MoE decode is memory-bandwidth-bound, so SMT siblings only
contend for the same core's load ports without adding bandwidth"，且
"the spin-barrier degrades badly when oversubscribed"。
`requested > 0` 时先铺满物理核、再铺剩余逻辑 CPU。

**本机实测拓扑**：`nproc = 10`，Intel Xeon E5-2666 v3，1 个 NUMA 节点（CPU 0-9），
**`cpu0`–`cpu9` 的 `thread_siblings_list` 各自只有自己** → **SMT 是关的**。
**所以这条在本机是空操作**：10 个逻辑 CPU 就是 10 个物理核，没有兄弟可避。

**唯一可保留的检查**：源码提到的"过度订阅会让自旋屏障退化"。
FastLLM 是 4 条 rank 线程（每卡一条）+ 主线程，跑在 10 个核上，**没有过度订阅**。

---

## 5. 建议的优先级（数字栏只放实测值）

| 优先 | 做什么 | FreeToken 依据 | 本机上限（计算） | 实测 |
|---|---|---|---|---|
| **1** | 把吐字窗口那 2352 次小拷贝融合成每步一次 | `offload_cache.py:380-443` | **12.0 ms / 59.2 ms 显卡空闲 = 20.3%**；占吐字窗口 8.6% | **未做、无数据** |
| 2 | 先把默认配置（CUDA Graph 开）下的真实空闲量出来 | FreeToken 也用 `engine/graph.py` 的 `GraphRunner` | — | **图开比图关快 10.26%**（已测，`/tmp/G_*.out`） |
| 3 | 给 `--low_gpu_mem` 的流式权重加 pin-after-fill | `host_banks.py:130-175` | 无上限数字可算（要先量 pageable vs pinned 的主机侧读延迟） | **未做、无数据** |
| — | 线程绑物理核 | `cpu_executor.py:92-143` | **不可用**：本机 SMT 已关 | 已核实（拓扑） |

**注**：优先 1 的 12.0 ms 上限是在**逐步发射路径**下测的；而 §10.3 已证明
**默认配置（CUDA Graph 开）下这条路径的大部分空闲并不存在**。
所以**动优先 1 之前，必须先做优先 2**——否则是在优化一个默认配置下不存在的病理。

---

## 6. 实现结果：优先 2 已测，优先 1 与 3 的前提被推翻

### 6.1 先说前提（每条都写下来）

- 优先 1（融合 2352 次小拷贝）与优先 3（流式权重 pin-after-fill）**共用一个前提**：
  **"吐字阶段主机侧是限制方"**。
- 优先 3 还多一个前提：**"`--low_gpu_mem` 下有权重从主机流式读取"**。

### 6.2 优先 2 实测（非侵入，1 Hz `nvidia-smi` 采样，2000 token 长解码）

**为什么要换仪器**：先试了 nsys，**去掉 `--cuda-graph-trace=node` 也仍然关掉图重放**
（那次 TPOP = 19.33 ms/token，而图开的正常跑是 12.19）。**所以 nsys 不能用来量默认配置**——
上一轮那 42.6% 的空闲是"图被仪器关掉"后的状态。改用看门狗包住、另加 1 Hz 非侵入采样。

| 臂 | TPOP | Total | sha | 加载完后的平均 GPU 利用率 | 对应空闲 |
|---|---:|---:|---|---:|---:|
| **图开（默认）** | **12.25 ms** | 27.6156 s | `7a76e196` | **90.5%** | **9.5%** |
| 图关 | 16.63 ms | 36.3730 s | `7a76e196` | **73.0%** | **27.0%** |

（产物：`/tmp/U_on.samples`、`/tmp/U_off.samples`、`/tmp/U_{on,off}.out`；两臂 sha 一致。）

**所以默认配置下的真实空闲是 9.5%，不是 42.6%。**

### 6.3 结论：优先 1 与优先 3 按原样不成立

**优先 1（融合小拷贝）**：图开时那 2352 次拷贝会被**捕获进图**，主机侧不再逐步发射，
而它们的数据量只有约 3 MB/解码窗口——**主机侧那 18.45 ms 的开销在默认配置下不存在**。
按 §6.2 实测，默认配置只剩 9.5% 空闲，**这条不earn 它的位置**。

**优先 3（流式权重 pin）**：**前提是错的。** `--low_gpu_mem` 在 C++ 侧**一处都没有**
（`grep lowGpuMem src include` = 0 命中），它只关掉 CUDA embedding 与 GPU token handoff；
**权重是显存常驻**（27B NVFP4 ≈ 12.6 GB，四卡各约 3.1 GB，实测各卡占 11.3 GB 含 KV）。
走主机权重那条路是**另一个开关 `--low`**（`SetLowMemMode`，`src/fastllm.cpp:483`），
本会话所有测量都没用它。

**唯一还站得住的残余**：`--low_gpu_mem` 会把 **embedding 表留在主机**
（`qwen3.cpp:2672`：`useCpuEmbedding = !GetCudaEmbedding() || GetLowMemMode()`）。
但解码每步只查 1 个 token 的若干行，命中的页极少。

### 6.4 若仍要做，适用条件是什么

| 原条目 | 什么条件下才成立 |
|---|---|
| 优先 1 融合小拷贝 | 只有在**必须逐步发射**的场景才有意义（挂 profiler 分析、动态形状导致图捕获失败）。此时空闲 27.0%，融合拷贝的理论上限是那 12.0 ms 的一部分 |
| 优先 3 pin 流式权重 | 只有切到 **`--low`**（权重驻主机）才成立；本会话未测过该配置 |

---

## 7. 实现结果：异步提交后重叠（submit → 排 GPU 活 → sync）——前提不成立

### 7.1 先写前提

FreeToken 那条模式（`moe/cpu_executor.py:543+` 的 `decode_submit`/`decode_sync`）要求：
**主机侧存在一块"与显卡接下来那步无关"的有用计算，可以塞进 submit 与 sync 之间。**
它在 FreeToken 里成立，是因为 FreeToken 把 **MoE 专家放在 CPU 上算**——那是一块几个核规模的计算。

**所以先做普查：解码期主机侧到底有多少 CPU 工作量。**

### 7.2 普查（非侵入，读 `/proc/<pid>/task/*/stat` 的 utime+stime）

**第一次测量（被污染）**：按"显存 ≥10 GiB"窗口统计得到 34.0 s 窗口内 CPU 128.15 s，
即 **377%（3.8 个核）**、每 token 64.08 ms。**这个数是错的**——那个窗口把
**CUDA Graph 捕获/实例化阶段**（CPU 密集）也算了进去。

**第二次测量（只取稳态解码的 12 s 窗口）**：

| 线程 | 状态 | 占核 |
|---|---|---:|
| 主 python 线程 | S | **0.50** |
| 其余 46 条线程合计 | S | 0.2 |
| **总计** | | **0.7 核 / 10 核** |

（产物 `/tmp/THR_on.threads`；那次 Total 27.6274 s、TPOP 12.26 ms。）

### 7.3 结论：不成立，不实现

**解码期主机只烧 0.7 个核（共 10 核），没有可重叠的独立计算。**
FreeToken 那条模式要重叠的是"CPU 专家计算"，而 FastLLM 这个配置下**没有 CPU 计算后端**。

**两块拼图里，FastLLM 其实已经有一块**：采样路径**已经在用固定主机内存**
（`src/blocks/llmsamplingblock.cpp:115-116` 的 `FastllmCudaHostMalloc` 给 hostIds/hostScores）。
缺的那块（独立 CPU 计算）在本配置下不存在，所以补不上。

**同一前提下的另外两条也被实测否掉**（§6）：融合小拷贝（默认配置空闲仅 9.5%）、
流式权重 pin（`--low_gpu_mem` 不流式权重）。
**三条同源条目共用一个前提"吐字阶段主机侧是限制方"，而这个前提已被两次独立测量否掉。**
