# 历史缓存（`--cache_history`）：为什么关掉它

**收口日期** 2026-09-18　**对象** FastLLM / Qwen3.8-27B-QUASAR-NVFP4 / 4×V100 TP4 / 生产 HTTP 服务

---

## 结论

**生产服务端已关闭 `--cache_history`。** 两条独立的理由，任一条都足够：

1. **代码上它根本没接上。** 写缓存的入口只挂在旧引擎里；生产跑的是新引擎，那个入口一次都不会被调用。缓存永远是空的，每次查询都落空。
2. **接上了也没活干。** 把分页前缀缓存压到池子容量的 1.34 倍之后，续写**仍然 100% 命中**（16000 个 token 全部复用），一次未命中都没出现。历史缓存能省的正是这种未命中。

**证据分级**（全文每条结论都标）：**实测** = 在目标系统上跑出来的；**观测** = 直接读代码或日志看到的；**计算** = 由观测按明确公式算出、未实测；**引自** = 别人系统里已记录的事实，带出处。

---

## 一、问题：这个功能为什么毫无效果

### 1.1 调用链（观测，行号均为 `src/models/basellm.cpp` 当前版本）

```
请求收尾点 → ResponseContext::TryRecord (:613) → basellm::TryRecordResponseContext (:617)
             → 过门 saveHistoryChat && UseGenericHistoryCache() (:625)
             → pastKVCacheManager.Record (:626)
```

本机型两道门都为真（运行日志 `saveHistoryChat=1 走通用历史缓存=1`），所以只要 `TryRecord` 被调用，缓存就会被写。

### 1.2 断在哪里

| 引擎 | 收尾点调什么 | 行号 |
|---|---|---|
| **新引擎** `RunNewMainLoop`（:1438 起）—— **生产在跑这个** | `ctx->TryRecordPagedCache(model)`（写**前缀**缓存） | :2912、:2918、:2938、:2944 |
| 旧引擎内联循环（:3016 到 :3457）—— 没在跑 | `it.second->TryRecord(model)`（写**历史**缓存） | :3100、:3398、:3403、:3417 |

**`TryRecord` 全仓库只有这四个调用点，全在旧引擎里。** 新引擎在同样的四个位置调的是前缀缓存，历史缓存的写入在移植时被漏掉了。

另一条写入口 `AddPromptCache`（:3674）只被 `tools/src/pytools.cpp:1105` 调用，是老的 Python 接口，与 HTTP 服务这条路无关。

### 1.3 运行日志佐证（实测）

三次请求的多轮对话测试，服务端日志：

| 事件 | 次数 |
|---|---|
| `查询入口` | 6（3 次请求 × 2 行） |
| `查询 -> 未命中` | 3 |
| **`记录入口`** | **0** |
| **`记录 新增`** | **0** |
| 同一次运行里 `[prefixcache] record OK: cachedLen=16000` | 1 |
| 同一次运行里 `[prefixcache] query 查表命中` | 2 |

同一批请求、同样四个收尾点：**前缀缓存记进去了，历史缓存一条都没有。** 这排除了"那些收尾点根本没被走到"这个替代解释。

**引擎选择的实证**：日志里 `[histcache] 引擎选中 GPUMainLoop`，即 `CanUseGPUForward()` 为真时走的 `RunNewMainLoop(true)`。

---

## 二、实测：接上了，这个功能有没有活干

**脚本** `tools/test_prefix_eviction.py`，不需要改任何引擎代码——命中情况由服务端返回的 `cached_tokens` 直接给出。

| 段 | 做什么 | 结果（实测） |
|---|---|---|
| **M0 对照** | 单会话 首轮 → 续写 | 首轮 6.786 s `cached=0`；**续写 0.221 s `cached=16000`** |
| **M1 施压** | 56 个互不相同的 16K 会话，并发 8 | 成功 **56/56**，失败 0，367.2 s，推入 **896000 token** |
| **M2 复测** | 再发一次会话 0 的续写 | **0.223 s `cached=16000`** |

**三条判据（跑前定死，全过）**：C1 三段跑完（56/56）；C2 对照命中（证明这把尺子测得出真假）；C3 推入量 896000 > 池子 668288（真的构成压力）。

**怎么读**：M2 和 M0 的续写耗时几乎一样（0.223 s 对 0.221 s），**比冷启动的 6.786 s 快 30 倍**，服务端同时报 `cached_tokens=16000`。不是碰巧快，是真的复用了那 16000 个 token。

**结论**：推入 896000 token（池子的 1.34 倍）之后，16000 token 的前缀仍然完整可复用，**一次未命中都没发生**。

### 这个实验没有证明的（边界）

- **没有直接观测分页缓存内部是否发生过驱逐。** 服务端 `--max_batch 8`，同时活跃的只有 8 个会话，其余请求结束后其页可能被释放而非保留。所以"压力是否真的落在前缀缓存上"这一条**未验证**。
- 无论哪种情况，对决策的影响相同：**在能造出的这个压力下，续写没有落空过。**
- **要更强地逼出驱逐**：`K_SESSIONS` 是环境变量，调大重跑即可；或改成让 8 路以上长会话**同时**保持活跃。

---

## 三、量级（计算，未实测）

一条 16K 会话的记录有多大：

| 组成 | 算法 | 大小 |
|---|---|---|
| 标准注意力 KV（随序列增长） | 16 层 × 2(K,V) × 4 KV头 × 256 维 × 1 字节(fp8) = 32 KiB/token；16384 token | **512 MiB** |
| 线性注意力状态（与序列长度无关） | 48 层 × (48 值头 × 128 × 128 × 4 字节 float32) = 48 × 3.0 MiB | **144 MiB** |
| **合计** | | **≈ 656 MiB** |

层级构成来自 `config.json` 的 `text_config.layer_types`：64 层中 16 层 `full_attention`、48 层 `linear_attention`（`full_attention_interval = 4`）。

**条数上限**：`PastKVCacheManager::maxRecordNum` 默认 5（`include/models/basellm.h:100`），`SetMaxRecordNum`（`basellm.cpp:866`）**全仓库无调用者**，也没有命令行参数能改 → 装满约 **3.2 GiB** 主机内存（本机共 30 GB，可用 21 GB）。

**搬运时间**：512 MiB 走 PCIe，按本机实测单向上限 13.16 GB/s 算 ≈ **39 ms / 条**。当前可页内存 + 同步拷贝下会更差，**未实测**。

---

## 四、如果要做，现在的搬运路径长什么样

`PastKVCacheMemory` 构造（`basellm.cpp:839`）对 64 层逐层做两件事：

1. `CopyFrom`（:850、:851）→ 内部先 `ToDevice(ori.dataDevice)`（`src/fastllm.cpp:1096`）→ **在显存里新分配一块、再做一次显卡到显卡的拷贝**
2. 若 `GetHistoryCacheInCPU()`（:853）→ `ToDevice(DataDevice::CPU)`（:854、:856）→ `src/fastllm.cpp:2816-2828`：`new uint8_t[]` 分配**可页内存**，再调 `FastllmCudaCopyFromDeviceToHost`（`src/devices/cuda/fastllm-cuda.cu:5921`），那是**同步 `cudaMemcpy`**，拷完立刻释放显存那块

**观测到的四个问题**：

| # | 问题 | 位置 |
|---|---|---|
| 1 | 同步搬运，整条计算流被堵住 | `fastllm-cuda.cu:5922`（`cudaMemcpy`，不是 async） |
| 2 | 目标地址是可页内存，即便换 `cudaMemcpyAsync` 也不会真异步 | `fastllm.cpp:2818`（`new uint8_t[]`） |
| 3 | 先白做一次显卡到显卡拷贝，再搬去主机 | `fastllm.cpp:1096` + `:2821` |
| 4 | 全程持着 manager 的锁 | `basellm.cpp:882` → `:927` |

**一条记录要发起 64 次显存分配 + 64 次显卡到显卡拷贝 + 64 次显卡到主机拷贝。** 这个形态正是业界明确记录过的"对卸载最致命的"碎片化，见 6.4。

---

## 五、仓库里已经有的积木（观测）

这套方案**不需要新写显卡端代码**。下面这套"非阻塞拷贝流 + 锁页内存 + 双槽 + 事件互等"的写法，仓库里已经在跑（`FastllmCatBatch` 的指针搬运）：

| 需要的能力 | 仓库里的现成实现 | 位置 |
|---|---|---|
| 非阻塞拷贝流 | `cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking)` | `fastllm-cuda.cu:16340` |
| 锁页主机内存 | `cudaMallocHost` | `fastllm-cuda.cu:16350` |
| 显卡 → 锁页主机，异步 | `FastllmCudaCopyFromDeviceToPinnedHostAsync(dst, src, size, stream)` | `fastllm-cuda.cu:5927` |
| 拷贝流等计算完成 | `cudaStreamWaitEvent(copyStream, kernelDone[slot], 0)` | `fastllm-cuda.cu:16375` |
| 计算等拷贝完成 | `cudaStreamWaitEvent(cudaStreamPerThread, copyDone[slot], 0)` | `fastllm-cuda.cu:16392` |
| 双槽轮换 | `pointerBuffer.slot ^= 1` | `fastllm-cuda.cu:16372` |
| 主机侧锁页标记 | `Data::isPinned` | `include/fastllm.h:605` |
| 计算的流 | `cudaStreamPerThread` | `fastllm-cuda.cu:411`（cuBLAS 也绑在这条） |
| 主循环的空闲时刻 | `if (seqLens.size() == 0) { ... wait ... }` | `basellm.cpp:2963` |
| 分页缓存回收页 | `ReleasePageIndex` / `ReleasePageIndices` | `include/fastllm.h:752-753` |

---

## 六、业界怎么做（引自，带出处）

### 6.1 都不等 GPU 空闲

| 系统 | 是否等空闲 | 靠什么不拖慢生成 |
|---|---|---|
| vLLM | **不等** | 专用流 + 计算流栅栏 + 提交推迟到下一个引擎步 |
| SGLang | **不等** | 两条固定流 + **逐层**事件握手 |
| TensorRT-LLM | **不等** | 两条独立流 + 步边界两次双向栅栏，主机全程不阻塞 |
| LMCache | **不等，但默认不重叠**（见 6.2） | 见 6.2 |

vLLM 的 RFC 原话：*"we will try to hide the transfer latency with async transfer (i.e. pinned memory, MemcpyAsync and/or separate streams)"*（https://github.com/vllm-project/vllm/issues/16144 ）。SGLang 官方文档：*"HiCache overlaps layers by concurrently loading the KV cache of layer N+1 while computing layer N"*（https://docs.sglang.io/docs/advanced_features/hicache_design ）。

**四家里唯一要求"系统空闲"的操作**是 SGLang 的 L3 存储后端热挂载/卸载：要求没有任何在跑、也没有任何在排队的请求，否则返回 HTTP 400（https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hicache_runtime_attach_detach.mdx ）。它等的是**请求级空闲**、只发生在**管理员改配置**时、**不在数据路径上**。

**"等空闲"在业界的位置就是：重配置的约束，不是日常搬运的约束。**

### 6.2 但 LMCache 的默认配置并不重叠

LMCache 有两代实现并存，网上很多说法混着讲：

- **进程内（旧，官方已标废弃）**：跑在引擎工作进程里，CPU 那层用 `cudaHostAlloc` 一次性预分配。
- **多进程（新，官方推荐）**：LMCache 是独立进程（`lmcache server`），CPU 那层是 POSIX 共享内存 + `cudaHostRegister` 事后锁页。

| 模式 | 存 | 取 | 重叠吗 |
|---|---|---|---|
| **默认**（`use_layerwise=false`） | forward 跑完后一把存，且 `store_stream.synchronize()` **强制同步、堵住 forward 线程** | `start_load_kv` 在 forward 之前，同步 `load_stream.synchronize()`，**堵住预填** | **不重叠** |
| `use_layerwise=true` | 每层算完发那层拷贝，用生成器把控制权交回引擎 | 逐层，只在该层要用的时刻等 | **重叠** |
| 多进程模式 | forward 线程只做 O(1) 登记，拷贝交后台线程池 | 同 | 重叠 |

**`use_layerwise` 默认是 `false`** —— LMCache 自己都没敢默认开。打开后官方文档说"算第 N+1 层时搬第 N 层"，论文实测端到端快 **1.46 倍**。

LMCache 的主机内存是一整块 `cudaHostAlloc` 预分配锁页池（默认 5.0 GB，**每 rank 一份**）；键是 `CacheEngineKey = 模型名 + TP规模 + rank + 这段前缀的哈希 + 数据类型`（不是块号，块号每步都在变）；存储单位 chunk = **256 token**，与引擎块大小解耦。

### 6.3 必须等计算流，否则输出乱码

vLLM 记录过：把 KV 搬到 CPU 时**没有先等计算流**，拷到了还没写完的 KV，脏数据后来被命中加载回去，**输出直接变成乱码**。消融实测（同一负载，n=456 请求，"hard garble" = 出现 ≥3 种非拉丁文字）：

| 配置 | hard garble |
|---|---|
| 卸载**关** | **0** |
| 卸载开 | 21–29 |
| 卸载开 + 强制拷贝**同步/有序** | **0** |
| 换用老的 `OffloadingConnector` | **0** |

最后两行证明根因就是跨流同步缺失（https://github.com/vllm-project/vllm/issues/45704 ）。主路径上同一个坑由 https://github.com/vllm-project/vllm/pull/31341 修复，那条 PR 同时做了两件事：*"we move the offloading to start at the beginning of the next engine step... Lastly, we remove the use of stream priorities as they don't effect DMA-based copies."*

**两条可直接照搬**：① 拷贝前必须在计算流上记事件、让拷贝流等它，**不能靠流优先级代替**；② 把 store 的**提交**推迟到下一个调度步开头，成本为零。

### 6.4 对本项目最要紧的一条：碎片化

vLLM 记录：*"This fragmentation is meaningless for model computation performance, but is **devastating for KV offloading**."*（https://github.com/vllm-project/vllm/issues/27742 ）改成"一个物理块装下所有层"后，物理块从 16 KB 变 2 MB，卸载吞吐**提升一个数量级**；仅"考虑 k/v 与头维度减少碎片"这一步就有 **4 倍**。

**对照第四节的观测**：我们现在是 64 层各拷一次、还多一轮显卡到显卡拷贝，正是这个形态。

**布局选择的影响（引自 SGLang 实测，https://github.com/sgl-project/sglang/pull/21631 ）**：16k 上下文下 `page_first` **1.890 ms/请求**、`page_first_direct` **39.765 ms/请求**，**差 21 倍**。

**分块粒度要按方向分别设计**：加载方向按"能被计算流水消费的粒度"切（SGLang 的 H2D 是逐层），写回方向按"能让 DMA 高效工作、能就地暂存"的粒度切（SGLang 的 D2H 是**每 64 页一块**）。两个方向不必同粒度。

**重叠的实现机制（SGLang）**：forward 走到第 N 层、去取那层 K/V buffer 时才在**当前计算流**上 `wait_event(load_events[N])`（`memory_pool.py :: MHATokenToKVPool.get_key_buffer`），所以拷贝流可以一路跑到第 N+1 层。**这是流级等待，不阻塞主机线程。**

### 6.5 写入策略：三种都有，默认不是"立刻写"

| 系统 | 策略 | 出处 |
|---|---|---|
| vLLM 主路径 | write-through，但**提交推迟一步** | https://github.com/vllm-project/vllm/pull/31341 |
| vLLM 第二套 | `lazy_offload=false` 立刻写 / `true` 只在**快被驱逐**时写 | `vllm/v1/simple_kv_offload/manager.py` |
| SGLang | 三档，由 `--hicache-write-policy` 决定：`write_through`（**默认**）/ `write_through_selective` / `write_back` | `python/sglang/srt/mem_cache/hiradix_cache.py` |
| LMCache 多进程 | 默认 `EVICTION_AWARE`：等持有 KV 的显存块**快被淘汰时**才提交 | `docs/source/mp/lazy_offload.rst` |

### 6.6 写回式（等驱逐才写）必然要付的代价

SGLang 的 `write_back` 模式下**确实会阻塞在拷贝上**，位置很关键（`writing_check(write_back=True)`，只在 `_evict_write_back` → `flush_staged` 这条路上被调用）：

```python
if write_back:
    # blocking till all write back complete
    while len(self.ongoing_write_through) > 0:
        for ack in self.cache_controller.ack_write_queue:
            ack.finish_event.synchronize()
```

**语义是"这些显存块要被复用了，所以必须等在飞的拷贝读完它们"**，和 vLLM 的 `jobs_to_flush` 一样是数据依赖。但**时机**很糟：写回发生在淘汰时刻，而淘汰时刻正是急着要拿回显存的时候。**等驱逐才写，等于把等待压在内存压力最大的那一刻。**

**注意引用版本**：SGLang 生产默认跑 `UnifiedRadixCache`，`HiRadixCache` 是 legacy（`registry.py` 默认链兜底是 `_create_unified_radix_cache`，整个文件无 `HiRadixCache` 的 import）。上面的代码来自 legacy 文件，**但同一条结论已在默认路径上核实到两个调用点，门控条件相同（`write_policy == "write_back"`）**。

**而且这条会阻塞的路径，SGLang 作者自己标了将来要废弃**（`_evict_write_back` 的 docstring：*"...otherwise drop them. **note this path will be deprecated in the future.**"*）。趋势是从"写回 + 淘汰时阻塞"移向"提前写 + 异步"，往 vLLM 那边收敛。

### 6.7 真正的判据：单层搬运时间 ≤ 单层计算时间

引自 CachedAttention 的 `Tload·Lhist > Tpref·Lnew`，以及 OrbitFlow 从 2K 到 16K 时 1.9 倍→6.1 倍的恶化曲线、Strata 在 75% 带宽下仍有 24% 停顿。**这个前提在长上下文下会崩。**

**套到本机（计算，未实测）**：16K 记录的 512 MiB 摊到 16 层 = 每层 32 MiB，按实测上限 13.16 GB/s 算 **单层搬运约 2.4 ms**；本机实测解码 17.7 ms/token 摊到 64 层是**每层约 0.28 ms**。**单层搬运比单层计算贵约 8.7 倍**，按层做流水线在这个尺寸上藏不住传输。

**收益侧（计算）**：命中时恢复 512 MiB 约 39 ms，重算 16000 token 的预填是 **6.786 s（实测）**——恢复比重算便宜约 **175 倍**。**收益是真的，前提是"真的会发生未命中"，而第二节的实测说没有。**

### 6.8 两家的共同空白（对本机是真实风险）

两轮研究都**明确没找到**（≠ 不存在）：

1. 没有一条记录说明"主机内存是可页内存会让 `cudaMemcpyAsync` 静默退化成同步"——只有 SGLang 一句机制描述讲的是小张量元数据，不是 KV 块本身。
2. **没有任何一条实测记录说明"卸载拷贝会和集合通信抢 PCIe / NVLink 带宽"。**

第 2 条对本机要紧：本机此前实测 **NCCL all-reduce 占内核时间 49.4%**，PCIe 单向已用掉上限的 **83.0%（余量 17.0%）**。卸载拷贝走同一条 PCIe。**这条没有现成结论可引用，只能自己测。**

### 6.9 其它已验证的坑

| 坑 | 内容 | 出处 |
|---|---|---|
| 锁页内存按 2 的幂取整 | PyTorch 的 host allocator 把每次分配向上取整到 2 的幂，**55 GiB 请求实占 84 GiB**；另一处 4 GiB 请求实测常驻 8.01 GB（1.87 倍） | https://github.com/vllm-project/vllm/issues/56410 |
| `cudaHostRegister` 大内存卡住启动 | TB 级时"单次调用跑很多分钟"；实测 512 GB 时 health 就绪 236.6 s → 139.0 s | https://github.com/vllm-project/vllm/issues/42632 |
| 可页内存的隐式同步 | *"Passing pageable host tensors ... still requires CUDA to stage the source through pinned memory. That staging can block the scheduler thread"* | https://github.com/sgl-project/sglang/pull/35944 |
| DMA 小页断崖 | H100 实测：4 KiB–24 KiB 平在 ~5–7 GB/s，**28 KiB 处跳 6 倍**到 27.5 GB/s | https://github.com/vllm-project/vllm/pull/42212 |
| 用显卡代码代替 DMA 更差 | 0% 命中率下"拷贝核"方案**比完全不开卸载还差 6%**（抢计算单元）；且核数不是越多越好，最小切片反而赢 | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| **默认值本身带已知风险** | SGLang 默认 `--hicache-io-backend=kernel`，但那个 I/O 核与 FA3 之间发生过 **illegal memory access**（就在显卡到主机拷贝与 forward 之间），部署被迫退回 `direct` | https://github.com/sgl-project/sglang/pull/8991 |
| **主机内存压力 → TP 下死锁** | 主机内存吃紧时各 rank 的写回数量会分叉，**只要因此跳过那次 all_reduce，NCCL 调用序列就错位，TP>1 直接死锁**。本机是 TP4 | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py |
| 两个系统锁页方式相反 | vLLM 默认 `torch.zeros(pin_memory=True)`，只在共享 `/dev/shm` 时才 `cudaHostRegister`；SGLang 在 CUDA/ROCm 上默认"匿名 mmap + 显式 `cudaHostRegister`"。注册失败报错值得引以为戒：*"host buffer is not pinned and device transfers may silently return stale data"* | `python/sglang/srt/mem_cache/pool_host/common.py` |
| LMCache：锁页池**每 rank 一份** | `max_local_cpu_size` 是每 rank 预算，TP8 设 512 GB 就是向主机要 **4 TB**，启动即 OOM，**日志里没有一行说过它乘了 8** | https://github.com/LMCache/LMCache/pull/5127 |
| LMCache：锁页对象泄漏 | lookup 命中但没被取回的对象永远钉住、**永远不会被淘汰**；用户报"跑几小时后 TTFT 从 0.4 秒变 3 秒" | https://github.com/LMCache/LMCache/issues/2017 |
| LMCache：开了它反而让解码变慢 | H100 单卡 Llama-3.1-8B 实测：**每 token 耗时 65.39 → 89.53 毫秒（+37%）**、输出吞吐 −40%。维护者称"即使零复用也不该有这么大开销" | https://github.com/LMCache/LMCache/issues/1326 |
| LMCache：多卡每次取回多付一次 PCIe | TP8 + H20 + MLA + 8192 token：leader rank 取回 **9.42 ms**，其余 rank 0.48 ms；修好后 0.70 ms，TTFT 133 → 123 ms | https://github.com/LMCache/LMCache/pull/3413 |
| LMCache：池满时无限忙等 | 淘汰不出候选时 `sleep(0.1)` 死循环，"引擎完全无响应" | https://github.com/LMCache/LMCache/issues/2942 |
| LMCache：锁页有硬上限 | `cudaHostAlloc` 受 `ulimit -l`（RLIMIT_MEMLOCK）限制，超了直接分配失败 | LMCache 文档 `offload_kv_cache.rst` |

---

## 七、三个方案与收益表

### 方案 A：空闲时才搬

收尾点只登记（不拷贝），主循环走到空闲点（`basellm.cpp:2963`，`seqLens.size() == 0`）时搬。

- 优点：改动最小，不需要流和事件。
- 缺点：拷贝期间主循环**仍被堵住**（不能注册新请求、不能开始下一轮解码）；队列里的 KV 要一直占显存；持续有流量时空闲点可能很久不出现。
- 与业界的关系：**四家都不这么做**（6.1）。

### 方案 B：独立流异步搬（业界路线）

收尾点立刻发起：**先在计算流上记事件、让拷贝流等它**（第 6.3 节证明这一步省不得），用 `FastllmCudaCopyFromDeviceToPinnedHostAsync` 拷进预分配的锁页内存池，再在拷贝流上记事件。主循环用 `cudaEventQuery` 轮询完成、不阻塞；完成后才挂进 manager、才释放显存。

- 优点：主循环和计算都不被堵；与下一轮解码重叠；忙时也能写。
- 必须做对的四件事：① 计算流栅栏（少了它 → **输出乱码**）；② 拷贝流用 `cudaStreamNonBlocking`；③ **推迟释放**（现在 `ToDevice(CPU)` 拷完立刻释放，异步下不能这么做）；④ 一个明确的阻塞栅栏，只在"显存要被复用时"使用。
- 零成本的习惯：把 store 的**提交**推迟到下一个调度步开头（6.3）。
- 缺点：要管锁页内存池、事件、延迟释放。

### 方案 C：写回式，平时根本不搬

收尾点什么都不做，等分页缓存真要回收那些页（`ReleasePageIndices`）时才搬。

- 优点：正常流量下搬运次数接近 0。
- 与业界的关系：SGLang 的 `write_back`、vLLM 的 `lazy_offload=true`、LMCache 多进程的 `EVICTION_AWARE` 都是这个策略。
- 缺点一：等驱逐才写，**把等待压在内存压力最大的那一刻**（6.6）。
- 缺点二（推断，未实测）：本机池子 668288 token、单会话 16K，正常流量挤不动它 → **在本机会退化成"永远不写"，也就是今天的现状。**
- 缺点三：引入"主机内存压力 → 各 rank 状态分叉 → TP 死锁"这条因果链，本机 TP4 正好踩在上面。

### 收益表

数字栏按项目规约只填实测值。三个方案**都还没做，因此都是"未做、无数据"**。

| 方案 | 改动量（估） | 实测数字 | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|---|
| A 空闲搬 | 小 | 未做、无数据 | 主循环存在空闲时刻 | 忙时空闲点不出现、队列涨 | 未做 |
| B 独立流异步 | 中 | 未做、无数据 | 锁页池、计算流栅栏、延迟释放都到位 | **漏掉计算流栅栏会输出乱码** | 未做 |
| C 写回式 | 中 | 未做、无数据 | 分页缓存**真的会发生驱逐** | 本机挤不动 → 可能退化成永远不写；TP4 下有跨 rank 分叉死锁风险 | 未做 |

**已经实测到的、与本决策直接相关的两条事实**：

1. 当前 `--cache_history true` 下，历史缓存记录数恒为 0（三次请求，`记录入口` 0 行）→ **这个功能今天收益为零。**
2. 同一批请求下前缀缓存正常命中（`[prefixcache] record OK: cachedLen=16000`，续写 TTFT 从 7.10 s 降到 0.58 s）。

---

## 八、当前生产状态与回退

生产服务端**已关闭 `--cache_history`**：

- `ftllm-server.service` 的 ExecStart 里只剩 `--prefix_cache true`；
- 为它加的两个日志环境变量（`FASTLLM_CACHE_HISTORY_TRACE` / `_LINES`）也一并撤掉——功能关了，日志只剩噪声（每个请求会打 4 行 `查询入口`）。

核对方式（都是权威来源，不是看配置文件）：

| 检查 | 结果 |
|---|---|
| 进程真实命令行 `/proc/<pid>/cmdline` | 只有 `--prefix_cache`，无 `cache_history` |
| 服务自报的启动参数 | `'cache_history': ''`、`'prefix_cache': 'true'` |
| 进程环境 | `CACHE_HISTORY` 相关变量 0 个 |
| 重启后发一个真实请求 | 正常返回，日志里 `[histcache]` 行数 = **0** |

**回退方式**：备份在 `/root/ftllm-server.service.before_off.bak`。把 `--cache_history true` 加回 ExecStart 即可；`src/models/basellm.cpp` 里的日志代码仍在（默认关闭，由环境变量控制）。

**如果以后要翻案**：把 `K_SESSIONS` 调大、或让 8 路以上长会话同时保持活跃（第二节标了这是本次没做到的），逼出一次真正的未命中。**那一次未命中就是这功能的开工条件。**

---

## 九、更正记录（我写错过的，逐条留痕）

| # | 原说法 | 错在哪 | 现状 |
|---|---|---|---|
| 1 | "`--cache_history` 多占约 183 MB 主机内存" | 和计算出的 656 MiB/条 差 3.6 倍；且当时历史缓存恒为 0 条记录，进程 RSS 里本就不含任何历史缓存数据。原数来自对比开关前后的 RSS 差值，没排除运行噪声就把差值归给了它 | **作废**，重新实测前不要引用 |
| 2 | "LMCache 在主机内存不是锁页时会静默降级、慢约 20 倍" | **这个 20 倍是 Intel XPU 专属**：XPU 那套代码里没有锁页分配的实现，静默退回可页内存，显卡到主机的存储掉到约 1.5 GB/s；补上后回到 10–30 GB/s。CUDA 侧也有静默降级（6 处只打 warning），但**没有这个数字** | **撤回**，改为"XPU 上 20 倍，CUDA 上无此数字" |
| 3 | "打开 `--cache_history` 后续写从 7.04 s 降到 0.605 s" | 归因错误。关掉 `--cache_history` 后是 0.594 s，两个配置的 cached 都是 16000。**收益来自 `--prefix_cache`，与历史缓存无关** | 已更正 |
| 4 | "关掉前缀缓存后，`cache_history` 独自也没有效果" | 当时把"没效果"归因成机制问题。真实原因是**写入路径根本不存在**（第一节） | 已更正，并找到了确切代码位置 |
| 5 | 引擎选择追踪里打出 2 行"旧引擎(内联循环)" | 变量默认值泄漏：它只在真正创建线程时才被赋值，第 2、3 次请求打印的其实是默认值 | 已修（默认值改 `nullptr`，四个分支各自显式赋值，只在真创建线程时打印）。修之前那个"2"是假的 |
| 6 | 第一次给历史缓存加日志后"编译通过" | 编译**失败**了，我的输出过滤只匹配英文 "error"，而编译器输出的是中文"错误"。是靠对比 `.so` 时间戳和抓二进制字符串才发现的 | 已修（辅助函数从第 771 行移到文件前部的匿名命名空间），并改用退出码判断 |
| 7 | 验证脚本"启动了" | 第一版把所有语料丢给取词器做二分查找，每个会话都编码一遍几 MB 的字符串，**40 秒烧了 44 秒 CPU、显卡一次没碰、输出一个字都没有**（Python 块缓冲还把这个失败藏住了） | 已修（先切 12 万字符窗口），单条从"无限慢"降到 0.82 秒 |

---

## 十、复现方式

```bash
# 1) 验证"这个功能有没有活干"（不需改引擎代码，约 8 分钟）
cd /home/fastllm
DRYRUN=1 PYTHONPATH=/home/fastllm/build-sm70-tests/tools python3 tools/test_prefix_eviction.py   # 干跑自检
tools/gpu_watchdog.sh /tmp/prefix_eviction.out --timeout 1200 --label prefevict -- \
  env PYTHONPATH=/home/fastllm/build-sm70-tests/tools python3 -u tools/test_prefix_eviction.py
grep -aE "M0 |M1 |M2 |判据|结论" /tmp/prefix_eviction.out      # 只看汇总行，不要 cat 原始日志

# 2) 看历史缓存的运行日志（需要先把 --cache_history 和 FASTLLM_CACHE_HISTORY_TRACE 打开）
journalctl -u ftllm-server.service --since today | grep histcache

# 3) 看服务自己报的启动参数
tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value ftllm-server.service)/cmdline
```

**相关文件**：`tools/test_prefix_eviction.py`（本轮的验证脚本）、`tools/test_history_cache3.py`（多轮对话命中测试）、`src/models/basellm.cpp`（历史缓存日志代码，默认关闭）、`/etc/systemd/system/ftllm-server.service`（生产配置）。
