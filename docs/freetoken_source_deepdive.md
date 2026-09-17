# FreeToken 源码深读：它到底怎么把"算"和"搬"叠起来

对象仓库 `/tmp/FreeToken`，浅克隆，commit `cac247a860e316e06580d05aeb05f2e647bde214`，版本 0.1.3。
本文所有 `path:line` 都是我在磁盘上按当前文件逐行核对过的。每条非平凡结论都标了 `[read from source]`（读源码得到）或 `[my inference]`（我的推断）。

单引号里的行号按文件本体现状编号。

---

## 0. 结论先说

FreeToken 把"搬权重"和"算权重"叠起来，一共三条独立的路子，各自用一对 CUDA event 做接力棒，不共用机制。

1. **视觉编码器的 block 双缓冲**（`models/weight_stream.py`）。一块在算，另一块从 pinned 主机内存搬进显存。缓冲只有 2 份，按 `i % 2` 轮转。这条跟 MoE 无关，只服务多模态编码塔。
2. **MoE prefill 的整层专家双缓冲**（`moe/offload_cache.py` + `layers/moe.py`）。GPU 算第 i 层专家的同时，拷贝流把第 i+1 层整层搬进另一块缓冲。缓冲 2 份，直接从显存里的统一 slot cache 前 `2 * num_experts` 个槽借出来。
3. **decode 期的 CPU/GPU 协同**（`moe/cpu_executor.py` + `layers/moe.py`）。同一层里，CPU 线程池算一部分专家，GPU 一边走 PCIe 抓另一部分、一边算。两侧靠 GPU 前端的内存操作（memop）握手，不用 kernel 轮询。

关于你说的三条，有三处要纠正。

- `weight_stream.py` 不是"prefill 权重流"，它是**视觉编码塔**的 block 流式加载，跟 MoE、跟 LLM 的 prefill 都不是一回事。它确实发生在 prefill 阶段（图像编码在 prefill 时做），但流的是 ViT block，不是 MoE 专家。`[read from source]` 见 `python/freetoken/models/qwen3_vl/vision.py:241-246`、`python/freetoken/models/gemma4/vision.py:205-210`。
- MoE 的 prefill 双缓冲**不是自有的两块显存缓冲**，而是从统一 slot cache 头部借 `2 * num_experts` 个槽（`python/freetoken/moe/offload_cache.py:611-616`）。这意味着 prefill 用的缓冲和 decode 用的 slot cache 是同一块内存，所以必须做失效处理，不能两块并存。
- 第 3 条更精确地说不是"CPU 跟 GPU 同时算门控"，而是**同一层的不同路由分配**给 CPU 和 GPU。每条路由只会被算一次，靠"GPU 侧权重清零 + CPU 侧 id 置 -1"这两个手段保证不重复算。

---

## 1. 仓库身份与内核构建方式

### 1.1 基本身份

| 项 | 值 | 出处 |
|---|---|---|
| commit | `cac247a860e316e06580d05aeb05f2e647bde214`（短 `cac247a`） | `git log -1` |
| 提交说明 | `chore(release): 0.1.3 (#489)`，2026-09-15 | `git log -1` |
| 版本 | 0.1.3 | 同上 |
| 包名 | `freetoken` | `pyproject.toml:10` |
| 定位 | 本地 MoE offload 推理运行时，带 OpenAI/Anthropic 兼容 API | `pyproject.toml:13` |
| Python 下限 | >= 3.10 | `pyproject.toml:15` |
| torch 区间 | `torch>=2.11,<2.12`，注释说明 PyPI 的 torch 2.11.0 wheel 本身就是 cu130 构建 | `pyproject.toml:54-58` |
| 主语言 | Python。`python/freetoken/` 下 600 多个文件，GPU 算子一部分是 Triton，一部分是 TVM-FFI 的 C++/CUDA 源码 | `[read from source]` 文件树 + `pyproject.toml:62` 的 `triton==3.6.0` |

### 1.2 编译型扩展（C++/CUDA）

`setup.py` 只声明了三个 **CppExtension**（不是 `CUDAExtension`），全部只用 host 侧 C++：

- `freetoken.kernel._pinned_tensor`，源文件 `python/freetoken/kernel/csrc/pinned_tensor.cpp`，`libraries=["cudart"]`，`-O3 -std=c++17`。`setup.py:42-51`
- `freetoken.kernel._cpu_moe`，源文件 `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`（2160 行），`libraries=["cudart"]`，`-O3 -std=c++17 -pthread`。注释说明：链 cudart 是为了 `cudaLaunchHostFunc` 的图节点；bf16 GEMV 用 per-function target attribute（`avx512bf16` / `avx512f`）+ 运行期 `__builtin_cpu_supports` 分派，不设全局 `-march`。`setup.py:52-66`
- `freetoken.kernel._ple_store`，源文件 `python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp`，仅 Linux 构建（`sys.platform == "linux"` 判断）。`setup.py:67-76`

`cmdclass` 用 `BuildExtension.with_options(use_ninja=True)`。`setup.py:78`

`_pinned_tensor` 是**必需项**。CUDA_HOME 缺失时直接抛错，错误信息写明"因为这扩展要链 CUDA runtime API"。`setup.py:23-33`

### 1.3 GPU kernel 怎么编

GPU 侧 kernel **不走 setup.py**，走两套机制并存。

**AOT 预编译成 kernel-cache wheel。** 包 `freetoken-kernel-cache/`（wheel 名 `freetoken_kernel_cache`），自带 `build_backend.py`，`pyproject.toml` 里 `packages = ["freetoken_kernel_cache"]`、`package-data` 含 `jit_cache/**/*.so`。`freetoken-kernel-cache/pyproject.toml`

AOT 的 kernel 清单在 `python/freetoken/kernel/aot.py:146-162` 的 `default_kernel_specs()`。跟本文直接相关的两条：

```python
specs.append(_fast_index_copy_multi_spec(num_threads=1024, blocks_per_bank=8))
# prefill hit-D2D gather (HBM-bound: wide grid) + its miss-side batch H2D binding.
specs.append(_fast_index_copy_multi_spec(num_threads=1024, blocks_per_bank=64))
specs.append(_batch_memcpy_spec())
```

`python/freetoken/kernel/aot.py:155-158` `[read from source]`

注意这里预编译了 `blocks_per_bank` 的两个取值 8 和 64，分别对应 decode 抓取和 prefill hit-D2D 收集（后者用 64，理由在 `offload_cache.py:776-777` 的注释里）。

**JIT 兜底。** 覆盖不到的变体由 `load_jit` 现场编，需要 nvcc。`install.sh:28-29` 的注释写明这一点，`install.sh:192-194` 在找不到 nvcc 时只 warn 不 die。

**安装流程。** `install.sh` 是 wheel 式的。建 venv → 装 `"${WHEEL}[accel]"` + flashinfer 预编译包 + `$KERNEL_CACHE_WHEEL`（`install.sh:214-236`）。所以普通用户装的是预编译产物；只有源码构建路径才会真的编 kernel（`install.sh:96-117`）。

**kernel 缓存的版本校验。** `kernel/utils.py` 里 `_kernel_cache_version_ok` 要求 release 部分相同，且两侧都带 `g<sha>` 戳时必须一致，否则拒绝加载。`python/freetoken/kernel/utils.py:99-114` `[read from source]`

---

## 2. 三条 overlap 机制逐个拆

### 2.1 机制一：视觉编码塔的 block 双缓冲（`BlockWeightStreamer`）

文件 `python/freetoken/models/weight_stream.py`，共 122 行。

**它服务谁。** 五个多模态家族都用它：`gemma4/vision.py:205-210`、`glm5_next/vision.py:105-109`、`minimax_m3/vision.py:110-114`、`muse_glimmer/vision.py:131-135`、`qwen3_vl/vision.py:244-248`。开关是 `--mm-encoder-weights`，`choices=["gpu","host"]`，默认值来自 `MultimodalConfig.encoder_weights`，而该字段默认是 `"host"`。`server/args.py:489-495`、`mm/config.py:19-21` `[read from source]`

所以**默认路径就是流式**。`server/args.py:495` 的 help 里写了数字，即"about 60 MiB of VRAM instead of the whole tower"。

**槽位数：2 份，为什么。** 类文档字符串一句话说清设计，`weight_stream.py:1`：

> "Block weights streamed from pinned host banks through two device staging buffers: block i computes from buffer i % 2 while the copy stream fills the other with block i + 1."

`[read from source]`

- 主机侧一个 bank，形状 `(num_blocks, row_bytes)`，uint8，pinned。`weight_stream.py:57`
- 设备侧 `staging` 形状 `(2, row_bytes)`，uint8。`weight_stream.py:58`

为什么是 2 而不是 1 或 3。1 份就没法叠（读和写同一块必须串行）；2 份正好够"当前算的 + 下一个在搬的"，而流水线的下一级（算 block i+1）必须等 block i 算完才能开始写回同一 buffer，所以 3 份不会换来额外并行度。`[my inference]`

为什么必须是**统一行宽**。看 `row_bytes = self._layouts[0][1]`，然后 `assert all(nbytes == row_bytes for _, nbytes in self._layouts), "streamed blocks must share one layout"`。`weight_stream.py:54-56` 也就是说该塔里每个 block 的参数量必须完全相同，否则这个 streamer 直接断言失败。`[read from source]`

**布局怎么算出来的。** `_slots()` 递归遍历 block 的 `__dict__`（跳过下划线开头的属性 `weight_stream.py:36-37`），把每个 Tensor 属性按 **256 字节对齐**（`_ALIGN = 256`，`weight_stream.py:12`）依次排布，累积偏移用 `offset += -(-nbytes // _ALIGN) * _ALIGN`。`weight_stream.py:33-45`

**pinning 发生在哪。** 就在构造函数的 `bank = torch.empty(..., pin_memory=True)`，`weight_stream.py:57`。注意这里用的是 torch 的 pin_memory 而不是自研的 `alloc_pinned_tensor`。搬完所有权重后有一次 `torch.cuda.synchronize(device)`，`weight_stream.py:62`，然后 `_bind_all_to_host()` 把每个 block 的每个属性重新绑成"主机 bank 里的那个视图"，`weight_stream.py:75-79`。原因写在注释里，是这样 `state_dict()` 仍然完整正确。`weight_stream.py:76`

**CUDA event 和 stream 清单。**

| 对象 | 创建处 | 谁 record | 谁 wait |
|---|---|---|---|
| `copy_stream` | `weight_stream.py:64` | 承载所有 `staging[buf].copy_` | 被 compute 流等 |
| `begin_event` | `weight_stream.py:65` | compute 流，`weight_stream.py:105` | copy 流，`weight_stream.py:106` |
| `ready_events[0..1]` | `weight_stream.py:66` | copy 流，`weight_stream.py:98` | compute 流，`weight_stream.py:111` |
| `release_events[0..1]` | `weight_stream.py:67` | compute 流，`weight_stream.py:116` | copy 流，`weight_stream.py:96` |

**fork-then-join 的确切顺序。** `blocks()` 生成器（`weight_stream.py:101-119`）：

1. `begin_event.record(compute)`，`weight_stream.py:105`
2. `copy_stream.wait_event(begin_event)`，`weight_stream.py:106`。注释写明原因：copy 流不许覆盖上一次 forward 还在读的 staging。`weight_stream.py:104`
3. 对每个 i：算 `buf = i % 2`（`:108`），`self._prefetch(i)`（`:109`），`self._prefetch(i + 1)`（`:110`），`compute.wait_event(self.ready_events[buf])`（`:111`），把该 block 的属性全部重绑到 `staging[buf]` 视图（`:112-114`），`yield i, op`（`:115`）。
4. yield 返回后（消费者已在本流上排好该 block 的算子）：`release_events[buf].record(compute)`（`:116`），`self._has_release[buf] = True`（`:117`），把属性重绑回主机 bank（`:119`）。

`_prefetch(i)` 内部（`weight_stream.py:88-99`）：

- `if i >= len(self._layouts): return`（`:89-90`），所以最后一层多出来的 `_prefetch(i+1)` 自然变成 no-op。
- `if self._holder[buf] == i: return`（`:92-93`），同一个 block 不重复搬。
- 进入 `torch.cuda.stream(self.copy_stream)` 上下文（`:94`）。
- `if self._has_release[buf]: self.copy_stream.wait_event(self.release_events[buf])`（`:95-96`）。这是**跨 forward 的复用保护**。
- `self.staging[buf].copy_(self.bank[i], non_blocking=True)`（`:97`），`self.ready_events[buf].record(self.copy_stream)`（`:98`），最后 `self._holder[buf] = i`（`:99`）。

**缓冲复用靠什么守。** 两个布尔数组。`_holder = [None, None]` 记录每个 buffer 现在装的是哪一块（`weight_stream.py:68`），用来做幂等；`_has_release = [False, False]`（`:69`）决定要不要等 release event。第一次 forward 时 `_has_release` 全 False，所以不会等一个从未 record 过的 event。

**调用方怎么插自己的计算。** 契约就是在 `for i, blk in self._blocks():` 的循环体里，直接在本流上排该 block 的算子。`qwen3_vl/vision.py:313-317` 就是这么用的，循环体里 `blk.forward(x, *attn)`，并且在 tap 命中时用 `groups[1 + k].copy_(..., non_blocking=True)` 把乘头结果直接搬回主机，`qwen3_vl/vision.py:316`。测试也验证了这个行为，连续两次 forward 结果相同，即第二次 forward 复用 staging，`tests/models/test_qwen3_vl_vision.py:69-71`。`[read from source]`

**没有的东西（重要）。** 这个文件里 `os.getenv` / `os.environ` 一次都没出现，我 grep 过（`grep -n "getenv\|environ" python/freetoken/models/weight_stream.py` 无输出）。所以它没有环境开关，只能通过 `--mm-encoder-weights` 整体开关。`[read from source]`

---

### 2.2 机制二：MoE prefill 的整层专家双缓冲

涉及 `python/freetoken/moe/offload_cache.py`（1077 行）和 `python/freetoken/layers/moe.py`（492 行）。

**缓冲即 slot cache 的前 2E 个槽。** 这是最容易误解的一点。`_init_prefill_overlap_buffers`：

```python
self.prefill_bank_buffers = [
    cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
    for _, cache in self.banks
]
```

`python/freetoken/moe/offload_cache.py:611-616` `[read from source]`

也就是说，每个 bank 一张 `[2, E, ...]` 的视图，`buffer[buffer_id]` 就是 `cache[buffer_id*E : (buffer_id+1)*E]`。`[my inference]`，依据是 `view(2, E, ...)` 是行优先的。

**槽位数：2，为什么。** 两个层面同时约束：

1. 概念上要 2 份，因为"当前层在算、下一层在搬"。`layers/moe.py:386-391` 的文档字符串：`_wait_prefill_overlap` "Double-buffer choreography ... kick off the next layer's full-layer H2D copy, then return this layer's bank views"。
2. 数量上必须 `cache_size >= 2 * num_experts`，否则构造就断言失败：

```python
assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
    "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
    "cache, so cache_size must be at least 2 * num_experts "
    "(raise moe_cache_size or disable moe_prefill_overlap)"
)
```

`python/freetoken/moe/offload_cache.py:163-167` `[read from source]`

预算侧的对应逻辑在 `engine/cache_budget.py:69-81`。`overlap = prefill_overlap and hi >= 2 * num_experts`，槽位下限 `lo` 在有 overlap 时是 `2 * num_experts`，否则 `num_experts`；而且如果最终 `moe_cache_size < 2E`，会把 overlap 再关一次（`cache_budget.py:80-81`）。`[read from source]`

**buffer_id = layer_id % 2。** `offload_cache.py:676`（`prefetch_prefill_layer`）、`offload_cache.py:826`（`wait_prefill_layer`）、`offload_cache.py:835`（`release_prefill_layer`）都是这一句。`[read from source]`

**CUDA event 和 stream 清单。**

| 对象 | 创建处 | 谁 record | 谁 wait |
|---|---|---|---|
| `prefill_copy_stream` | `offload_cache.py:618` | 承载整层非阻塞 H2D | 被 compute 流等 |
| `prefill_ready_events[0..1]` | `offload_cache.py:619` | copy 流，`:698` 或 `:816` | compute 流，`:829` |
| `prefill_release_events[0..1]` | `offload_cache.py:620` | compute 流，`:839` | copy 流，`:696` 或 `:790` |
| `prefill_begin_event` | `offload_cache.py:621` | compute 流，`:657` | copy 流，`:658` |

注意 `:698` 与 `:816` 是两个不同分支的 record 点（整层拷贝分支 / hit-D2D 切分分支），后面细说。

**fork-then-join 的确切顺序。** 入口是层自己的 `_wait_prefill_overlap`：

```python
def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
    if self.layer_id == 0:
        cache.begin_prefill()
    cache.prefetch_prefill_layer(self.layer_id)
    cache.prefetch_prefill_layer(self.layer_id + 1)
    return cache.wait_prefill_layer(self.layer_id)
```

`python/freetoken/layers/moe.py:392-396` `[read from source]`

跟机制一的关键差别在于，机制一在 `blocks()` 里对**每一层**都重新打一次 begin fence；机制二只在 `layer_id == 0` 时打一次。`[read from source]`

然后 `_prefill_routed` 拿着返回的 views 跑 GEMM，最后放行：

```python
if cache.prefill_overlap:
    views = self._wait_prefill_overlap(cache)
    out = self._expert_gemm(..., views=views, n=self.num_experts, alphas=cache.alphas_for_layer(self.layer_id), is_prefill=True)
    cache.release_prefill_layer(self.layer_id)
    return out
```

`python/freetoken/layers/moe.py:359-372` `[read from source]`

`begin_prefill` 干三件事（`offload_cache.py:645-666`）：

1. `_prefill_buffer_layer = [None, None]`、`_prefill_buffer_released = [True, True]`（`:648-649`）。这是每个 chunk 的复位。
2. `prefill_begin_event.record(torch.cuda.current_stream(self.device))`（`:657`），`prefill_copy_stream.wait_event(self.prefill_begin_event)`（`:658`）。
3. 如果 hit-D2D 可用，在 copy 流上把 `slot_for_id` 快照进 pinned 主机张量，并 `prefill_copy_stream.synchronize()`（`:664-666`）。注释说明这是"每个 chunk 一次主机同步"，之后每层的分类都是纯主机数学。`offload_cache.py:661-663`

第 2 步的注释值得完整引用，因为它是**跨 forward 的复用保护**的核心：

> "Fence this prefill's copy-stream work behind everything already enqueued on the compute stream. The release/ready events only order against the *previous prefill*; under overlap scheduling a new prefill can be enqueued while the preceding decode batch is still running, and that decode may have loaded experts into the slots the buffers borrow -- without this fence the first prefetch would stomp bytes a running GEMM is reading."

`python/freetoken/moe/offload_cache.py:651-656` `[read from source]`

`prefetch_prefill_layer`（`offload_cache.py:668-701`）的顺序：

1. `if not self.prefill_overlap or layer_id >= self.num_layers: return`（`:669`），最后多加的那一次 `prefetch(layer_id + 1)` 落在这里变成 no-op。
2. `buffer_id = layer_id % 2`（`:676`），`if self._prefill_buffer_layer[buffer_id] == layer_id: return`（`:677-678`）幂等。
3. 复用断言：`if self._prefill_buffer_layer[buffer_id] is not None: assert self._prefill_buffer_released[buffer_id], "Prefill overlap buffer is being reused before release"`（`:679-682`）。
4. 定义内部 `copy()`（`:684-687`）：先 `self._invalidate_prefill_buffer(buffer_id)`（`:685`），再对每个 bank `buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)`（`:687`）。
5. 三条分支：hit-D2D 走 `_prefetch_split`（`:689-690`）；没有 copy 流时裸调 `copy()`（`:691-692`）；否则进 copy 流，先 `wait_event(release_events[buffer_id])`（仅当 `_prefill_buffer_has_release_event[buffer_id]` 为真，`:695-696`），再 `copy()`，最后 `prefill_ready_events[buffer_id].record(self.prefill_copy_stream)`（`:698`）。
6. 收尾 `_prefill_buffer_layer[buffer_id] = layer_id`、`_prefill_buffer_released[buffer_id] = False`（`:700-701`）。

`wait_prefill_layer`（`offload_cache.py:818-830`）先再调一次 `prefetch_prefill_layer(layer_id)`（`:825`，幂等），`assert self._prefill_buffer_layer[buffer_id] == layer_id`（`:827`），然后 compute 流 `wait_event(prefill_ready_events[buffer_id])`（`:829`），返回 `tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)`（`:830`）。

`release_prefill_layer`（`offload_cache.py:832-841`）先做 `release_events[buffer_id].record(torch.cuda.current_stream(self.device))`（`:839`），`_prefill_buffer_has_release_event[buffer_id] = True`（`:840`），`_prefill_buffer_released[buffer_id] = True`（`:841`）。

**借槽带来的副作用，必须失效。** `_invalidate_prefill_buffer`（`offload_cache.py:635-643`）：

```python
slot_start = buffer_id * self.num_experts
old_ids = self.id_of_slot[slot_start:slot_end]
self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
old_ids.fill_(-1)
self.usage[slot_start:slot_end].zero_()
```

注释解释了 `usage` 清零的用意，是让这些槽变成最老的，`ensure_experts` 里 `argmin(usage)` 选受害者时会最先挑它们。`offload_cache.py:641-642` `[read from source]`

**hit-D2D 切分（可选，默认关）。** `moe_prefill_hit_d2d` 默认 `False`（`engine/config.py:49`，`engine/engine.py:1359`，`server/args.py:757-758`）。打开后 `_prefetch_split`（`offload_cache.py:746-816`）改成：

- 用 chunk 开始时的 `slot_for_id` 快照（pinned numpy）在**主机侧**分类 hit/miss，命中判据是 `snap >= 2 * E`（`:770-771`）。
- 命中的行在 **compute 流**上做 D2D 收集：`prefill_hit_compact`（一个 Triton 单 launch，把活的行压成固定形状的 gather 索引，`offload_kernels.py:63-82`）+ `fast_index_copy_multi_jit(..., blocks_per_bank=64)`（`:774-786`）。注释给出理由："HBM D2D needs the wider grid (~22 GB/s per 1024-thread block on H100)"，`:776-777`。并且说明"Serializing the gather before this layer's GEMMs costs its plain duration instead of nondeterministic SM contention"，`:750-751`。
- 未命中的行在 copy 流上走**一次** `cudaMemcpyBatchAsync`（`:788-816`）。
- 小于 `_SMALL_BANK_FEAT_BYTES`（256 KiB）的 bank，整层当一个 entry 搬，**即使零 miss 也搬**，理由在 `:798-801`："it keeps every batch entry above the driver's async floor and covers the hit rows the gather skips for these banks"。

那个 256 KiB 阈值的来由值得单独引用，它是一段实测结论：

> "cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0, empirically bisected: a single 5-22KB entry beside one large entry blocks the calling thread for the full transfer; >=253KB entries never do). A synchronous call still moves bytes at full PCIe rate but stalls the host, which un-hides the GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e)."

`python/freetoken/moe/offload_cache.py:17-25` `[read from source]`

`_hit_d2d_usable`（`offload_cache.py:703-733`）列出全部失效条件，任何一条不满足就永久回落到整层拷贝并**只警告一次**：快照或流未初始化（`:712-713`）、`FREETOKEN_SKIP_FAST_INDEX_COPY` 被设（`:714-715`）、fused copy plan 不可用（`:716-717`）、`cache_size <= 2 * num_experts` 导致没有 hit 区（`:718-722`）、`cudaMemcpyBatchAsync` 不可用（`:723-724`）。

**跟 unpinned 层互斥。** `set_bank_sources` 里：如果任何层是 LOCKED/PAGEABLE 且 `prefill_overlap` 开着，直接抛 `ValueError`，理由是"prefill overlap DMAs from registered banks"。`offload_cache.py:325-336`。engine 侧对应地在 split residency 时把 `moe_prefill_overlap` 强制改成 `False`（`engine/engine.py:615-621`），并打一行 INFO 解释。

**runtime 重建时会重估。** `rebuild()` 第一步先拆掉 prefill 缓冲（因为它们的视图指向旧的 `bank_caches`，`offload_cache.py:476-484`），第 5 步重新评估：如果新的 `cache_size < 2E` 就把 `prefill_overlap` 关掉并 warn，`offload_cache.py:526-534`。

**调用方怎么插自己的计算。** MoE 这边调用方不需要做什么，层自己把 GEMM 塞在 `wait` 和 `release` 之间。这就是它跟机制一的根本差别。机制一暴露一个生成器让调用方自己插，机制二把整段编排封在 `_prefill_routed` 里。

---

### 2.3 机制三：decode 期的 CPU/GPU 协同（hybrid）

**它是什么。** 同一层的一次 decode 里，把"这一步缺的专家"分成两半：一半走 PCIe 抓到 GPU slot cache 上由 GPU 算，另一半留给 CPU 线程池算。约定是每条路由只算一次。

**入口 `_decode_hybrid`（`python/freetoken/layers/moe.py:298-345`），逐行顺序：**

```python
raw = topk_ids.clone()                                    # :316 原始 expert id 留给 CPU
cache.ensure_experts_hybrid(self.layer_id, topk_ids)      # :317 就地改写为 slot 或 -1
if cache.collect_stats: cache.record_decode_stats_hybrid(self.layer_id)   # :318-319
on_gpu = topk_ids >= 0                                    # :320
cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()     # :322
pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)  # :323
cpu_routed_early = executor.decode_sync(pending) if not _HYBRID_OVERLAP else None     # :327-329
cache.copy_missing()                                      # :331 PCIe H2D 只抓 fetched 的那些
gpu_slots = topk_ids.clamp_min(0)                         # :332 -1 -> slot 0
gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()     # :333
gpu_routed = self._expert_gemm(cache, hidden_states, gpu_w, gpu_slots, views=cache.bank_views(), ...)  # :334-343
cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)  # :344
return gpu_routed + cpu_routed                            # :345
```

`[read from source]`

**"每条路由只算一次"靠两个互补手段。** 文档字符串写得很清楚（`layers/moe.py:311-312`）：

> "Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0)."

`[read from source]`

C++ 侧的跳过确实存在，我在多处读到 `if (e < 0 || e >= num_experts) return;` 或 `continue;`，例如 `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp:1596`、`:1663`、`:1695`、`:1741`、`:1766`、`:1829`。`[read from source]`

**fork-then-join 的确切顺序。** 这里没有第二根 GPU 流，全部在**当前 CUDA 流**上：

1. **fork**：`decode_submit` 在流上排 D2H 拷贝，然后排"叫醒 CPU"的节点，就返回，不同步。`layers/moe.py:323`
2. GPU 流继续往下排队：`copy_missing()`（`:331`）排 PCIe H2D，`_expert_gemm`（`:334`）排分组 GEMM。
3. **join**：`decode_sync`（`:344`）在流上排"等 CPU 完成"，然后排 H2D 结果回送。

所以"重叠"就是第 2 步的 GPU 工作跟 CPU 池的工作在时间上并行。`[read from source]` 依据是 `decode_submit` 的文档字符串（`cpu_executor.py:550-557`）：

> "Issue the D2H copies + the CPU-pool submit host node, then return without waiting. Lets a caller (the hybrid backend) enqueue GPU work between this and :meth:`decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch."

**握手协议（两条路）。** `cpu_executor.py:32-51` 的长注释和 `cpu_moe_ext.cpp:565-580` 的长注释是同一件事的两侧描述。

- **flag 路径（默认）**：`_FLAG_SYNC = os.getenv("FREETOKEN_CPU_MOE_FLAG_SYNC", "1") != "0"`，`cpu_executor.py:51`。GPU 侧在 submit 时做 `cuStreamWriteValue64`：先把 `done[slot]=0` 再把 `ready[slot]=1`。顺序在 C++ 注释里被点名解释："Order matters and is preserved by the front end: reset done BEFORE raising ready, so the coordinator's completion write for THIS step can never be wiped."，`cpu_moe_ext.cpp:649-651`（函数体在 `:646-656`）。sync 时做 `cuStreamWaitValue64(done[slot] >= 1)`，`cpu_moe_ext.cpp:659-664`。C++ 侧一个常驻协调线程 `coordinator_loop`（`cpu_moe_ext.cpp:2019` 起）轮询 `ready[]`、跑任务、写 `done[]`：轮询点在 `flag_load_acquire(&ready_flags[L]) != 0`（`cpu_moe_ext.cpp:2044`），消费后 `submit(t); sync();`，最后 `flag_store_release(&done_flags[L], 1)`（`cpu_moe_ext.cpp:2056`）。
- **host-func 路径（兜底）**：`cudaLaunchHostFunc` 排 submit/sync 两个主机回调，`cpu_moe_ext.cpp:1970-1977`。

选哪条在构造时定，即 `_flag_sync = _FLAG_SYNC and device.type == "cuda"`（`cpu_executor.py:206`），再用 `memops_probe` 做一次功能性探测（`cpu_executor.py:208-219`，C++ 实现 `cpu_moe_ext.cpp:620-628`，探测内容是 WRITE(7) + WAIT(>=7) + stream sync）。探测失败就打一行 INFO 退回 host-func。

**为什么不用 kernel 轮询。** `cpu_executor.py:39-43` 和 `cpu_moe_ext.cpp:573-578` 给了同一个理由，值得引用：

> "(The first cut used a spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU dynamic power schedulers responded by clamping the CPU frequency -- a net decode regression on power-coupled edge devices.)"

`[read from source]`

**flag 槽位数和为什么是 16。** `_FLAG_SLOTS_PER_LAYER = 16`（`cpu_executor.py:52-55`），容量 `num_layers * _FLAG_SLOTS_PER_LAYER`（`:275`），三个 pinned int64 数组 `_ready/_done/_err`（`:277-282`）。注释给的理由是"每个 (layer, decode batch size) 组合一个槽，覆盖这么多个不同 decode batch size（含 capture 的图尺寸和 eager padding 过的尺寸）；再多就没见过，溢出的组合只是退回 host-func 路径"。`cpu_executor.py:52-55` `[read from source]`

槽位在 `_task_for` 里懒分配，即 `slot = len(self._flag_slots)`，`if slot < self._flag_capacity:` 才注册，`cpu_executor.py:521-525`。

**pinning 发生在哪（这条机制里）。** 三种：

1. CPU 侧的 IO 缓冲，每个 batch size 一组，用 `alloc_pinned_tensor` 分配：`x`（bf16）、`ids`（int32）、`w`（float32）、`y`（bf16）。`cpu_executor.py:493-503`。
2. flag 数组，`alloc_pinned_tensor(..., dtype=torch.int64)` 后清零。`cpu_executor.py:277-282`。
3. 专家 bank 本身，由 bank 加载路径 pin（机制二/五里已经说明）。`cpu_executor._resolve_banks` 只是取 `data_ptr()` 建每层地址表（`cpu_executor.py:325-338`），并且把表和层张量都挂在 `self._banks` 上做 GC 保护，因为 C++ 持有裸指针（注释 `cpu_executor.py:195-196`）。

**缓冲复用靠什么守（这条机制里）。** `self._io: dict[int, dict[str, torch.Tensor]]`（`cpu_executor.py:259`）按 batch size 缓存；`self._tasks: dict[tuple[int,int], int]`（`:260`）按 (layer, batch size) 缓存 task 描述符。两者都是懒分配、然后复用，这正是能进 CUDA graph 的前提。`_init_cpu_moe_executor` 的文档字符串说明必须**在 graph capture 之前**建好执行器，因为 eager warmup forward 会把这些 pinned 缓冲和 task 指针实体化，后续 capture 的 host/memcpy 节点就嵌这些稳定地址。`engine/engine.py:742-748` `[read from source]`

**CPU 线程池怎么定。** `resolve_threads_and_affinity`（`cpu_executor.py:119-141`）的规则是，`requested == 0` 时一个物理核一个线程并绑核；显式数量则先铺满物理核再铺剩下的逻辑 CPU。理由写在 `physical_core_cpus` 的文档字符串里："MoE decode is memory-bandwidth-bound, so SMT siblings only contend for the same core's load ports without adding bandwidth."，`cpu_executor.py:93-99`。另外 flag 模式且自动线程数时，会把最后一个物理核留给协调线程（`cpu_executor.py:222-229`）。

**看门狗。** `_watchdog_main` 每 2 秒醒一次（`cpu_executor.py:663-675`），`_watchdog_tick` 的判死条件是三个 AND：doorbell 还挂着（`ready==1 && done==0`）、首次怀疑时就挂着、且期间 `flag_served_count` 没变（`cpu_executor.py:607-646`）。判死后先置 `_err[i] = 1` 再置 `_done[i] = 1`，`cpu_executor.py:642-645`，注释写"after err: unblock the stream into a checked failure"。`raise_if_unhealthy` 每次 forward 由 engine 调一次（`engine/engine.py:997` 附近），只读一个 pinned 值。

**关掉重叠做 A/B。** `_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"`，`layers/moe.py:26`。置 0 后 `decode_sync` 被提前到 `copy_missing` 之前（`layers/moe.py:327-329`），重叠窗口消失。注释明确说这是"measurement-only escape hatch"。`layers/moe.py:23-25`

**取多少 miss 走 PCIe。** 见第 4 节。相关 API 是 `ensure_experts_hybrid`（`offload_cache.py:855-874` → `offload_kernels.py:43-60`）。

---

## 3. `fast_index_copy` 的零拷贝 PCIe gather

### 3.1 GPU 怎么直接读主机 pinned bank

kernel 在 `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh`。多 bank 融合版的关键结构：

```cpp
struct MultiIndexCopyParams {
    const int64_t* __restrict__ dst_ptrs;     // [B] device, each base addr of a bank slot cache
    const int64_t* __restrict__ src_ptrs;     // [B] device, each GPU-visible base addr of a bank host source
    const int64_t* __restrict__ feat_bytes;   // [B] device, per-row bytes (multiple of 16)
    ...
};
```

`fast_index_copy.cuh:475-484` `[read from source]`

kernel 体里 `src` 就是主机 bank 的 base 地址，直接解引用：

```cpp
const auto* src = reinterpret_cast<const uint8_t*>(p.src_ptrs[b]);
auto* dst = reinterpret_cast<uint8_t*>(p.dst_ptrs[b]);
const uint4 v = *reinterpret_cast<const uint4*>(src + ps * feat + col);
*reinterpret_cast<uint4*>(dst + pd * feat + col) = v;
```

`fast_index_copy.cuh:495-511` `[read from source]`

grid 是 `blocks_per_bank * num_banks`（`fast_index_copy.cuh:559-560`），block 到 bank 的映射是 `b = blockIdx.x / kBlocksPerBank`（`:490`）。行内以 16 字节为单元拆分，`units = feat >> 4`（`:499`），注释写明 `feat % 16 == 0` 是前提。

**访存指令是特意的。** `load_nc` / `store_nc` 用内联汇编：

- 读：`ld.global.L1::no_allocate.b32/v2.b32/v4.b32`，`fast_index_copy.cuh:36-52`
- 写：`st.global.wt.b32/v2.b32/v4.b32`（wt = write-through），`fast_index_copy.cuh:54-71`

函数名里的 `nc` 就是 no-cache / non-coherent 的意思。`[my inference]`，依据是函数名和指令里的 `L1::no_allocate`、`wt`。这些指令绕开 L1 分配和 L2 缓存污染，对"流式抓一次性字节"是对的。

**主机地址怎么进 kernel 参数。** host 侧 `device_alias`（`fast_index_copy.cuh:149-159`）：

```cpp
inline void* device_alias(void* ptr, DLDevice dev) {
    if (dev.device_type == kDLCUDA || host_ptr_identity()) return ptr;
    void* mapped = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&mapped, ptr, 0);
    ...
}
```

`host_ptr_identity` 的判定是 `cudaDevAttrUnifiedAddressing == 1 && cudaDevAttrCanUseHostPointerForRegisteredMem == 1`（`fast_index_copy.cuh:135-147`）。在 Linux 上这两条都成立，所以主机的 host VA 就是 GPU 能解引用的地址，不需要翻译。在 Windows/WDDM 上映射到另一个设备地址，所以必须翻译。`[read from source]`

Python 侧 `device_ptr` 做同样的判断，另外缓存结果（`lru_cache(maxsize=1)`），理由是一个进程只 pin 一个 CUDA 设备：

```python
def device_ptr(t: torch.Tensor) -> int:
    if t.is_cuda or _host_ptr_identity():
        return t.data_ptr()
    return _load_pinned_extension().host_device_ptr(t.data_ptr())
```

`python/freetoken/kernel/pinned.py:59-68`，缓存装饰器在 `pinned.py:53-54`。`[read from source]`

### 3.2 pinned 分配路径

`python/freetoken/kernel/csrc/pinned_tensor.cpp`（128 行）暴露五个函数：

| 函数 | 实现要点 | 行 |
|---|---|---|
| `create_pinned_tensor_like` | 按 sizes+strides 精确算 nbytes（`:24-33`），`cudaMallocHost(&data_ptr, alloc_nbytes)`（`:37`），`torch::from_blob(..., free_pinned, options)`，options 带 `pinned_memory(true)`（`:41-43`） | `pinned_tensor.cpp:13-44` |
| `alloc_pinned_tensor` | `cudaHostAlloc(&data_ptr, alloc_nbytes, cudaHostAllocPortable \| cudaHostAllocMapped)` | `pinned_tensor.cpp:46-72`，分配点 `:61-62` |
| `host_ptr_identity` | 查 UVA 和 registered-mem 两个属性 | `pinned_tensor.cpp:77-85` |
| `host_device_ptr` | `cudaHostGetDevicePointer` | `pinned_tensor.cpp:87-95` |
| `host_register` | `cudaHostRegister(..., cudaHostRegisterPortable \| cudaHostRegisterMapped)` | `pinned_tensor.cpp:97-103` |

释放统一走 `free_pinned` → `cudaFreeHost`（`pinned_tensor.cpp:7-11`）。

**零拷贝为什么要求 mapped。** `alloc_pinned_tensor` 里 `cudaHostAllocMapped` 上方的注释写得很直接：

> "Portable + mapped: the offload gather kernel reads these banks straight from host memory (zero-copy), which requires device-mapped pinned pages."

`pinned_tensor.cpp:58-62` `[read from source]`

### 3.3 为什么不用 `torch.empty(pin_memory=True)`

`python/freetoken/kernel/pinned.py:1-6` 整个文件文档字符串就是回答这个问题的：

> "Exact-size pinned host tensors (e.g. offload expert banks).
>
> The offload gather kernel (`fast_index_copy`) reads host memory zero-copy from the GPU, so allocations must be pinned + device-mapped. We avoid `torch.empty(pin_memory=True)` because its caching allocator rounds sizes up to the next power of two (a 70GB bank would reserve 128GB)."

`[read from source]`

两件事同时成立才需要自己造：**必须 mapped**（否则零拷贝不成立），**必须精确尺寸**（否则 70 GiB 的 bank 会预留 128 GiB）。

**反例确认。** 仓库里 `torch.empty(pin_memory=True)` 仍然在用，但都用在尺寸可控的小张量上，例如 `models/weight_stream.py:57`（视觉塔 bank，尺寸是 block 数乘行宽）、`moe/offload_cache.py:624`（`[num_layers, num_experts]` 的 int32 快照）、`scheduler/scheduler.py:918`（`[3, needed]`）。`[read from source]` 也就是说避开它是因为**尺寸量级**，不是因为 torch 的 API 有问题。`[my inference]`

**bank 的 pin 是 pin-after-fill。** 专家 bank 走的是 `HostBank`（`moe/host_banks.py:78-169`），默认 backing 是懒匿名 mmap（`:112`），填完数据才 `pin()`（`:130-144`），`pin()` 里调 `host_register`（`:138-141`）。文件头注释算了一笔账：

> "Registering already-resident pages just page-locks them; registering a lazy mmap first faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then immediately overwritten by the read. So pin-after-fill removes a whole redundant pass."

`moe/host_banks.py:5-13`，数字在 `:9-10` `[read from source]`

pin 失败会被包成 `PinFailed`（`host_banks.py:142-143`），engine 捕获后给出针对 pin 预算的提示（`engine/engine.py:649-650`）。`[read from source]`

### 3.4 "position == expert id" 不变式

**先说什么叫 position。** 这里说的是"专家权重张量里第 p 行装的是哪个专家"。如果第 p 行就是专家 p，那么上层可以直接拿原始 routing id 当行索引，不用查表。

**这个不变式由 `materialize_layer` kernel 建立。** `python/freetoken/moe/offload_kernels.py:249-285`：

```python
tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)   # :280
tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)  # :281
tl.store(usage_ptr + slot, step, mask=expert_mask)              # :282
tl.store(evict_slots_ptr + off, slot, mask=expert_mask)         # :283
tl.store(src_indices_ptr + off, off, mask=expert_mask)          # :284  layer-local row
tl.store(num_indices_ptr, num_experts)                          # :285
```

`[read from source]`

因为 `slot = off`（`:266`）且 `off` 就是专家序号，所以 slot p 装专家 p，`slot_for_id[layer, p] = p`。第 284 行 `src_indices[off] = off` 配上第 283 行 `evict_slots[off] = off`，意味着随后的 `copy_missing` 也是"第 p 行从主机 bank 第 p 行搬到设备第 p 行"，恒等映射。

**双缓冲沿用同一性质。** `prefill_bank_buffers` 是 `cache[buffer_id*E : ...].view(E, ...)` 的视图（`offload_cache.py:613-616`），所以 buffer 内的第 p 行还是专家 p。`layer_id % 2` 只影响用哪块，不影响块内偏移。`[my inference]`，依据是 `view(2, E, ...)` 的行优先语义。

**所以 routing id 可以"原样透传"。** `_prefill_routed` 的文档字符串一句点到：

> "In both, position == expert id, so the routing ids pass through unmapped."

`python/freetoken/layers/moe.py:353-356` `[read from source]`

`_wait_prefill_overlap` 里再说一次，"buffer position == expert id, so routing ids pass through unmapped"，`layers/moe.py:389-390`。

**下游消费方全部按这个前提写。** 每处都明写：

- `layers/quantization/moe/nvfp4.py:192-196`，"full-layer prefill passes banks whose position == expert id (the materialized `[:E]` slot view or the overlap double buffer views), so the raw routing ids arrive unmapped"
- `moe/fused_nvfp4.py:300-302`，同样措辞
- `moe/fused_fp8_block.py:22`，"(``[num_experts, ...]``, position == expert id)"
- `kernel/triton/fp8_blockscale_moe.py:221-222`，"(materialized layer: position == expert id)"
- `layers/quantization/moe/fp8_block.py:59`，`n = view.n if view.n is not None else layer.num_experts`，即 prefill 时按 `n` 截前 n 行

`[read from source]`

**对比，decode 不满足这个不变式。** decode 走 `ensure_experts` 的 LRU 改写，`topk_ids` 被就地换成 slot id，可以是任意槽位，所以必须走 gather。`offload_cache.py:267-276` 的 `_decode_routed` 文档字符串说明了这一点，并且它调用 GEMM 时传 `n=None` 和 `alphas=cache.alphas_for_slots(...)`（`layers/moe.py:287-296`）；prefill 传 `n=self.num_experts` 和 `alphas_for_layer`（`layers/moe.py:361-370`）。`[read from source]`

两个 alphas 取法的差别就是不变式的直接体现。`alphas_for_slots` 需要通过 `id_of_slot` 反查（`offload_cache.py:576-586`），`alphas_for_layer` 是连续切片（`offload_cache.py:588-596`）。

**违反不变式会怎样。** `copy_missing` 对 unpinned 层有硬检查。如果是 LRU 改写而不是整层 materialize 就抛 `RuntimeError`，理由是"ensure_experts's LRU slot remap cannot be honored without a device alias"。`offload_cache.py:1015-1021`。另外 `is_unpinned_layer` 的文档字符串也说明这条路径"presumes materialize's position == expert id"，`offload_cache.py:571-574`。`[read from source]`

---

## 4. `q*` 拆分策略的实现实况

### 4.1 名字对不上

README 里写的是"bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy)"，`README.md:20`。

我在整个仓库里 grep 过 `q_star`、`qstar`、`q^\*`，源码里**没有任何符号叫 q\***。`[read from source]`

源码里能对上的只有两个标量：

1. 一个**后端选择的布尔判断**（hybrid 还是 offload），阈值默认 2.0。
2. 一个**每步取多少 miss 走 PCIe 的比例**（fetch fraction）。

把 README 的名称映射到这两个具体实现上是我做的，`[my inference]`。仓库没有给出从 $q^\star$ 到代码名字的推导。

### 4.2 公式一：hybrid 还是 offload

```python
def recommend(cpu_bw_gbs: float, pcie_bw_gbs: float, threshold: float = 2.0) -> str:
    """``hybrid`` iff CPU bandwidth exceeds ``threshold`` x PCIe bandwidth, else ``offload``."""
    return "hybrid" if cpu_bw_gbs > threshold * pcie_bw_gbs else "offload"
```

`python/freetoken/moe/benchbw.py:598-600` `[read from source]`

- 默认阈值 2.0：函数签名默认值 `benchbw.py:598`，`run_benchbw` 参数默认值 `benchbw.py:693`，CLI `--threshold` 默认值 `benchbw.py:951-952`，文档 `docs/cli.md:202`，模块文档字符串 `benchbw.py:17-19`。
- 严格不等式 `>`，所以正好等于 2 倍时判为 offload。
- 判定用的**是独立测量的数字**，`benchbw.py:662`：`entry["recommended"] = recommend(cpu_g, pcie_g, threshold)`，其中 `cpu_g = entry["cpu_moe_gbs"]`、`pcie_g = entry["pcie_gather_gbs"]`（`benchbw.py:659`）。
- 拿不到 CPU 路径（例如 block-fp8）或某个 bench 挂了，一律判 offload，`benchbw.py:636-637`、`:675-678`。

**运行期怎么用这个结论。** `bench_profile.load_backend_recommendation`（`bench_profile.py:114-153`）先把 `expert_quant` 映射到 bench 的格式 key（映射表 `bench_profile.py:25-31`：`mxfp4 -> mxfp4_triton`，其它同名）。优先读 per-dtype 结论（`bench_profile.py:133-137`）；没有就退到 per-model 结论，并且**要求全部一致才返回 hybrid**，只要有一个判 offload 就保守判 offload（`bench_profile.py:139-153`）。

engine 端的消费在 `engine/engine.py:1596-1626`。判定条件在 `:1596-1603`（`default_backend == "offload"` 且 profile 返回 `"hybrid"`），然后有两道闸。模型激活不被 CPU 扩展支持就留在 offload（`:1607-1612`），编译好的 `_cpu_moe` 扩展太旧也留在 offload（`:1613-1621`）；都过了才 `default_backend = "hybrid"`（`:1623`），最后 `override("moe_strategy", default_backend)`（`:1627`）。`[read from source]`

### 4.3 公式二：取多少 miss 走 PCIe

这才是真正决定"叠多少"的那个数。读侧 `bench_profile.py:156-191`：

```python
cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
if cpu_ov and pcie_ov:
    return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
if cpu and pcie:
    return min(1.0, pcie / cpu)
return None
```

`python/freetoken/moe/bench_profile.py:185-190` `[read from source]`

- **有"竞争下测量"的一对**：`fraction = pcie_ov / (pcie_ov + cpu_ov)`，最后 clamp 到 `[0, 1]`（`min(1.0, ...)`，`bench_profile.py:187`）。
- **只有独立测量的一对**：退化为 `fraction = pcie / cpu`，同样 clamp（`bench_profile.py:190`）。
- 两个都是 None 就返回 None（`bench_profile.py:191`）。

函数文档字符串给这两种情况的语义（`bench_profile.py:164-171`）：

> "The hybrid backend's bandwidth-matched fetch split: of a decode step's expert misses, fetch this fraction over PCIe and compute the rest on the CPU, so both finish together. Preferred source is the *overlapped* pair (CPU MoE and PCIe gather measured while running concurrently -- the real contention regime): fetched/misses = pcie_ov / (pcie_ov + cpu_ov). Older profiles without it fall back to the standalone bandwidths under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA), which reduces to pcie/cpu."

`[read from source]`

**工程侧的等价说法**在 `engine/engine.py:711-722`：

> "Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw : (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode step's misses"

`[read from source]`

注意这两个写法在代数上是一致的。若 `F/M = pcie/(pcie+cpu)`，则 `F : (M-F) = pcie : cpu`；而"CPU 在 DMA 占满时剩 `cpu - pcie`"给出 `F : (M-F) = pcie : (cpu - pcie)`，那 `F/M = pcie/cpu`。两句说的是同一件事的两种表述，前者用总 CPU 带宽做分母，后者用"被抢走一部分之后的净带宽"做分母。`[my inference]`

**整数取整规则（这个在 GPU 和 CPU 两侧必须一致）。** `offload_kernels.py:56-57` 先把 fraction 转成 Q16 定点：

```python
frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
```

然后在 kernel 里选让"较慢那一侧"最小的整数邻居，`offload_kernels.py:345-354`：

```python
lo = (num_missing * fetch_frac_q16) >> 16
cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
cost_hi = tl.maximum((lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16)
max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
```

注释说明理由，即"fetch time scales with F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick the integer neighbor that minimizes the slower (max) side of the overlap."，`offload_kernels.py:346-348`。`[read from source]`

CPU 参考镜像用同一规则（`offload_kernels.py:158-162`，`cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)`），测试要求两侧逐位一致（`tests/moe/test_hybrid_fetch.py:96-113`）。测试还固化了一个"不要用 ceil"的回归，即 `_balanced_fetch(3, round(0.415 * Q)) == 1`，理由写在 `tests/moe/test_hybrid_fetch.py:36-39`。`[read from source]`

**配置入口。** `moe_hybrid_max_fetch` 默认 `-1`（`engine/config.py:65`，`engine/engine.py:1357`），`-1` 表示 auto。engine 的解析在 `engine/engine.py:711-738`：显式非负值直接返回（`:719-720`），否则读 profile；读不到就把 `hybrid_max_fetch` 设成 1 并 warn（`:727-731`）；读到就把 `hybrid_max_fetch = cache.num_experts`（注释说"inert: the fraction is the cap"）并设 `hybrid_fetch_fraction = fraction`（`:735-736`）。`[read from source]`

### 4.4 独立测量 vs 竞争下测量

三个测量函数，各自测什么很明确：

| 函数 | 测什么 | 是否与对方同时跑 | 行 |
|---|---|---|---|
| `measure_cpu_mem_bw` | 主机 DRAM 读带宽上限（STREAM 式读） | 独立 | `benchbw.py:184-253` |
| `measure_pcie_bw` | 线性 pinned 到设备拷贝 H2D/D2H | 独立 | `benchbw.py:256-279` |
| `measure_pcie_gather_bw` | 真实 `copy_missing` 抓一整层 | **独立** | `benchbw.py:416-446` |
| `measure_cpu_moe_bw` | 真实 CPU MoE GEMV（bs=1） | **独立** | `benchbw.py:506-535` |
| `measure_overlap_bw` | 上面两个**同时**跑 | **竞争** | `benchbw.py:538-595` |

`[read from source]`

前两个是"天花板"，`benchbw.py:6-8` 的模块文档字符串这么叫它们，它们只出现在报告的 `ceilings` 字段里（`benchbw.py:790-794`），不参与任何决策。

后三个才是"real kernels"（`benchbw.py:382` 的分节标题）。谁进决策：

- `cpu_moe_gbs`（独立）和 `pcie_gather_gbs`（独立）决定 **hybrid/offload 判定**，`benchbw.py:659-662`。
- `cpu_moe_overlap_gbs` 和 `pcie_gather_overlap_gbs`（竞争）决定 **fetch split**，`benchbw.py:663-669`。原注释：

> "Both sides work standalone -> also measure them contended (concurrently). This pair sets the hybrid backend's fetch split (load_hybrid_fetch_fraction); the hybrid-vs-offload verdict above stays on the standalone numbers."

`benchbw.py:663-665` `[read from source]`

**竞争测量的具体做法**（`measure_overlap_bw`，`benchbw.py:538-595`）：先各自热身 8 轮（`:562-565`），然后 `threading.Barrier(2)` 同步起跑（`:568`）。CPU 侧在 worker 线程里循环跑 bs=1 CPU decode step，`run_task` 会释放 GIL（注释 `:551`）；PCIe 侧在主线程循环跑整层 `copy_missing`，并且**每次拷贝都 `torch.cuda.synchronize(device)`**（`:589`），注释解释"so the DMA is really in flight, not just enqueued"（`:551-552`）。两侧各报自己的字节数和自己的耗时，注释承认"the two windows differ by at most one CPU step + one gather"（`:553`）。`[read from source]`

**取整数时有个细节**：CPU MoE 的 step 函数会轮转路由窗口，让连续步访问不相交的专家集合，就是为了稳定在 DRAM-bound 区间而不是吃到 LLC 红利（`benchbw.py:467-472`、`:480-481`）。合成 bank 的内存预算 `_SYNTH_BANK_BUDGET = 2 << 30`（`benchbw.py:75`），理由在 `:319-324`："bandwidth is per-byte, so fewer experts don't change the GB/s"。

### 4.5 文档字符串给的"独立数字为何不能预测拆分"的确切理由

`measure_overlap_bw` 的文档字符串第二段（`python/freetoken/moe/benchbw.py:542-548`）原文：

> "The standalone numbers cannot predict this split. Assuming full DRAM contention (CPU keeps ``cpu_bw - pcie_bw`` under DMA) over-penalizes a CPU kernel that never saturated DRAM to begin with -- the DMA then mostly rides the leftover bandwidth; assuming no contention ignores it entirely. Measuring the contended pair directly gives the hybrid backend its bandwidth-matched fetch split: fetched : cpu-computed = pcie_ov : cpu_ov."

`[read from source]`

拆成三句人话：

1. 假设"DRAM 被 DMA 全占"（即 CPU 只剩 `cpu_bw - pcie_bw`）会**高估惩罚**，因为 CPU 那个 GEMV kernel 本来就没把 DRAM 打满，DMA 其实是蹭剩下的带宽，不是硬抢。
2. 假设"完全不竞争"则是彻底忽略。
3. 两种假设都不对，所以只能真跑一次同时测量的版本。测出来直接给 `fetched : cpu-computed = pcie_ov : cpu_ov`。

**并且"混合结论"要保守。** 如果 per-model 多条工作负载对同一格式给出不一致的判定，`load_backend_recommendation` 要求全票 hybrid 才返回 hybrid，否则 offload（`bench_profile.py:139-153`）。理由注释：`"a mixed verdict (a near-threshold format) resolves conservatively to 'offload'"`（`bench_profile.py:123-125`）。

**这个 bench 默认跑什么。** 无参数时不跑 per-model，跑 per-dtype"调参 bench"，因为格式决定一切（代码在 `benchbw.py:974-979`，理由在 `:21-27` 和 `:132-138`）。per-dtype 的几何用 `DTYPE_WORKLOADS`（`benchbw.py:139-146`）。结果落到 `$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json`（`bench_profile.py:39-46`），每张卡一个文件；profile 里的 GPU 名跟当前卡不一致就忽略整份 profile（`bench_profile.py:104-110`）。全局默认路径里的 `mmap` 语义和阈值定义都在本文件，没有别处重复定义。`[read from source]`

---

## 5. 重叠所依赖的支撑机制

### 5.1 全局 LRU 专家缓存

**数据结构。** `OffloadMoeCache.__post_init__`（`offload_cache.py:148-290`）里：

- `slot_for_id`，`[num_layers, num_experts]` int32，层级映射。`offload_cache.py:169-174`
- `id_of_slot`，`[cache_size]` int32，扁平 id 空间的反向映射。id 的编码是 `layer_id * num_experts + expert`，注释说明这样"一个数组就能代替 (layer, expert) 二元组，淘汰一个槽不需要解码"，`offload_cache.py:175-183`。
- `usage`，`[cache_size]` int64，时间戳 LRU 的计数器。`offload_cache.py:184`
- `step`，标量 int64。`offload_cache.py:185`
- `evict_slots` / `src_indices` / `num_indices`，一次待完成拷贝的槽位、源行号、数量。`offload_cache.py:187-191`。三者合起来就是"pending copy 状态"，由 `copy_missing` 消费。`offload_cache.py:263-268`

**`ensure_experts`（decode，GPU 路径）。** `offload_cache.py:843-853` 只做两件事：可选地记路由直方图（在 id 被改写之前，`:846-850`），记录 `_pending_src_layer` 和 `_pending_whole_layer = False`，然后调 `offload_kernels.ensure_experts`。

真正的实现在 `offload_kernels.py:19-40`，它**委托给 flashlib 的 `lru_ensure`**：

```python
lru_ensure(
    expert_ids, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage, cache.step,
    expert_ids, cache.src_indices, cache.evict_slots, cache.num_indices,
    stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
    id_base=layer_id * cache.num_experts,
)
```

`python/freetoken/moe/offload_kernels.py:28-40` `[read from source]`

三个契约在这段里被点名（文档字符串 `:20-27`）。`id_base` 做 layer 和扁平空间之间的翻译；`out_indices` 传的就是 `expert_ids` 自己，"preserving the in-place rewrite every downstream GEMM depends on"；`src_indices` 映射回本层的行号。

`flashlib` 是硬依赖，`pyproject.toml:40-41` 注释写明 "slot_cache: the device-side LRU admission kernel behind the MoE expert cache"。

**`ensure_experts_hybrid`（decode，混合路径）。** `offload_cache.py:855-874` 转给 `offload_kernels.py:43-60`，后者按 `expert_ids.is_cuda` 分流到 GPU kernel（`:97-125`）或 CPU 参考实现（`:128-188`）。文档字符串说明契约：最多把 `hybrid_max_fetch`（或 fraction 决定的）个 miss 分配槽位并排拷贝，溢出的 miss 就地改写成 `-1`；`num_indices` 是**封顶后**的 fetch 数（给 `copy_missing` 用），`num_missing_full` 是**封顶前**的 miss 数（给统计用）。`offload_cache.py:858-864`。`[read from source]`

GPU kernel 分三个 Phase（`offload_kernels.py:333`、`:372`、`:400` 的注释标题）：Phase 1 算 active 和 missing 并定 cap；Phase 2 按 `argmin(usage)` 选受害槽，只对封顶的那些做；Phase 3 把 `expert_ids` 重写成槽号或 -1。取哪些 miss 由 `BY_RECENCY` 决定（默认按 recency 降序、id 升序，`offload_kernels.py:364-368`；否则取最小 id，`:369-370`），开关是 `FREETOKEN_HYBRID_FETCH`（`offload_kernels.py:14-16`）。Phase 2 里有个细节：`owner_active` 掩码把"本步活跃专家占着的槽"的 usage 提到极大值，让它们不会被选为受害者（`:378-382`）。`[read from source]`

`expert_recency` 这个 `[num_layers, num_experts]` 数组在每次 active 后都更新（`:408-410`），注释解释意图，即"an overflow miss computed on the CPU now ranks high if it recurs, so it gets fetched next time"（`:406-407`）。

**`copy_missing`（三条件分支）。** `offload_cache.py:1011-1053`：

```python
layer_id = self._pending_src_layer
assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
if layer_id in self._unpinned_layers:          # :1015
    ...整层同页 H2D，:1024-1026，同步拷贝
    return
if self._copy_fused_ok:                        # :1027
    fast_index_copy_multi_jit(self._copy_dst_ptrs, self._copy_src_ptrs[layer_id],
                              self._copy_feat_bytes, self.evict_slots, self.src_indices,
                              self.num_indices)                                    # :1034-1041
    return
for per_layer, cache in self.banks:            # :1046-1053 逐 bank 回退
    fast_index_copy_jit(cache, self.evict_slots, per_layer[layer_id], self.src_indices, self.num_indices)
```

`[read from source]`

融合分支的注释解释了为什么能省 launch 数，原文是"One launch copies the missing rows for every bank (instead of one launch per bank). evict_slots/src_indices/num_indices are shared across banks; src_indices holds layer-local expert rows, resolved against this layer's source pointers (layer_id is a static int per captured graph node)."，`offload_cache.py:1030-1033`。

kernel 侧的原因说明在 `fast_index_copy.cuh:466-474`，那里给了规模感，原文是"the single-bank kernel above needs one launch per bank (e.g. 6 banks * 36 layers = 216 launches/decode step, all near-empty at a warm/full cache)"。`[read from source]`

### 5.2 融合多 bank 拷贝计划 `_build_fused_copy_plan`

`offload_cache.py:380-441`。它预计算三张 int64 设备张量，一次性建好、之后不再变，所以是 CUDA-graph 安全的（注释 `:385-386`）。

- `_copy_dst_ptrs`，`[B]`，各 bank 的 slot cache base 地址。`:423`
- `_copy_src_ptrs`，每层一张 `[B]`，该层各 bank 的主机源地址。`:424-427`
- `_copy_feat_bytes`，`[B]`，每个 bank 每行多少字节。`:428`

源地址用的是 `device_ptr(source)` 而不是 `data_ptr()`，注释说明，原文是"The kernel dereferences these on the GPU, so store each host bank's device alias (== data_ptr() under UVA identity; differs on Windows/WDDM)."，`offload_cache.py:413-416`。

**对齐要求，不满足就整体禁用。** 每行的字节数必须是 16 的倍数、每个 base 地址必须 16 字节对齐（`feat % 16 != 0 or cache.data_ptr() % 16 != 0` 就 `return`，`:406-407`），源地址同样要求（`:418-419`）。禁用后 `_build_copy_plan`（`:367-378`）会检查逐 bank 回退是否可行：如果某个 bank 的行字节数不是 128 的倍数就抛 `RuntimeError`，错误信息说明"只有融合路径能搬它们，但它被关了"。`offload_cache.py:371-378` `[read from source]`

**小 bank 的特例（hit-D2D 用）。** `_SMALL_BANK_FEAT_BYTES = 256 * 1024`（`offload_cache.py:26`），`_gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]`（`:434`），只有这些 bank 需要 D2D 收集，因为小 bank 走整层 H2D、其行根本不需要 D2D（注释 `:432-433`）。`:435-440` 做索引切分，`:441` 置 `_copy_fused_ok = True`。

**重建时必须重算。** `rebuild()` 第 3 步重新分配 slot cache 后立刻重调 `_build_copy_plan()`，注释写明"slot caches were reallocated -> refresh fused-copy addrs"，`offload_cache.py:499`。这也是为什么地址能当常量用：缓存生命周期内地址不变，重建时全部重来。

**`blocks_per_bank` 的两个取值。** decode 抓取用默认 `8`（`offsets_cache.py` 无关，是 `kernel/fast_index_copy.py:162` 的默认参数 `blocks_per_bank: int = 8`），prefill hit-D2D 用 `64`（`offload_cache.py:785`）。文档字符串给了完整推导（`kernel/fast_index_copy.py:164-178`），核心两句：

> "The copy is host->device PCIe-bandwidth bound (~31 GB/s measured, vs ~3 TB/s HBM), so what saturates the link is the TOTAL in-flight request count, blocks_per_bank*num_threads (~4096 threads/bank is the knee; blocks and threads are interchangeable -- 8x1024 == 16x512 == 4x1024 all measure the same across small/medium/large-expert workloads)."

`python/freetoken/kernel/fast_index_copy.py:166-169` `[read from source]`

这里两个数字（~31 GB/s、~3 TB/s）是仓库自己写下的，我只做引用，不替它背书。

### 5.3 FTW 快速权重格式

文件 `python/freetoken/checkpoint/ftw.py`，666 行。

**身份。** 格式标签 `freetoken_weight`（`ftw.py:51`），版本 1（`:52`），对齐 4096（`:53`），默认分片上限 8 GiB（`:54`）。

**它解决什么问题（模块文档字符串 `ftw.py:1-31`）。** 两点：

1. **对齐**。每个张量起始落在 4096 对齐的区域偏移上并补齐到 4096，分片也在 4096 边界切。所以任意张量（或它在分片内的任意切片）都能做到"偏移、长度（向上取整到 4096）、目标地址"三者同时块对齐，这正是 O_DIRECT 的要求。`:13-17`
2. **统一**。同时装稠密权重（`kind="weight"`，喂给 `load_state_dict`）和 offload 专家状态（`kind="experts_bank"`，即后端重排后的按专家 bank）。`:18-23`

写者 `FTWWriter`（`ftw.py:126-204`），读者 `FTWReader`（`ftw.py:208-358`），流式产出 `iter_ftw_weights`（`ftw.py:365-432`），bank 重建 `load_ftw_banks`（`ftw.py:435-659`）。

**为什么它跟重叠有关。** 三个直接接口：

1. **O_DIRECT 直读到已有的 pinned bank**。`read_into` 把张量的逻辑字节范围映射到一个或多个分片文件范围，然后分块多线程 O_DIRECT 直读进目标缓冲，不经过页缓存。`ftw.py:320-358`。O_DIRECT 不支持时（tmpfs、overlay、网络挂载）退到 `mmap` 而不是分块缓冲读，理由在 `ftw.py:226-231`："a whole-shard mapping + kernel readahead copies far faster than per-chunk page-cache reads"。
2. **按层拆分，让 pin 和读并行**。这是给双缓冲铺路的：per-layer 布局时每个 `(bank, layer)` 是独立 FTW 条目的 `f"{bank_name}#L{layer_id:05d}"`（`ftw.py:61-67`、`layer_bank_entry_name:64-67`），因而每层可以独立 pin。`load_ftw_banks` 用 `PinPipeline` 在后台线程边读边 settle（`ftw.py:570-596`），注释说明："Jobs are per (bank, layer) -- many small reads, so a wider pool; each bank pins as its read completes, overlapping cudaHostRegister with the remaining reads."，`ftw.py:566-567`。`PinPipeline` 本身的注释给理由："cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle)."，`moe/host_banks.py:287-289`。
3. **bank 已经是内核要的行布局**。`load_ftw_banks` 直接产出"每层一个 `[num_experts, ...]` HostBank"的 per-layer 契约（`ftw.py:439-441`），`set_bank_sources` 直接吃它（`offload_cache.py:292-365`），中间没有重排步骤。

**两种磁盘行布局。** 文档字符串在 `ftw.py:446-462`：flat region（一个 bank 一个条目，一整块 `[num_layers*num_experts, ...]`，层内字节范围通常不是 4096 对齐，所以要用"对齐的包围窗口"读进 page-aligned scratch 再切视图，`:447-458`）和 per-layer（一个 `(bank, layer)` 一个条目，起始已对齐，直读，`:459-462`）。两种混在同一 name 上会被断言拒绝（`ftw.py:516-517`）。

**residency 的落地。** `layer_residency` 默认全 pinned（`ftw.py:473`）。PINNED 走 `cudaHostRegister`（或 born-pinned 的 `cudaHostAlloc`，见 `ftw.py:476-482`），LOCKED 走 mlock（省 pin 配额），PAGEABLE 留在普通 mmap。实际结果会回写到 `ExpertBanks.layer_residency`，并且 mlock 失败会降级为 PAGEABLE 并记日志（`ftw.py:621-651`）。`[read from source]`

### 5.4 CUDA graph 兼容性

**哪些是图内。** `GraphRunner`（`engine/graph.py:105-232`）。判据很简单：`can_use_cuda_graph` 只在 decode 且 `batch.size <= max_graph_bs` 时为真，`graph.py:204-205`。所以**所有 prefill 都不在捕获范围内**。捕获循环在 `graph.py:173-198`，每个 batch size 先跑一次 eager forward，再用 `torch.cuda.graph(...)` 捕一次（`:189-194`）。

**捕获前必须先备好的东西。** 执行器必须在 capture 前建好，因为 warmup forward 会实体化 pinned 缓冲和 task 指针，捕获的 host/memcpy 节点嵌的是这些稳定地址。`engine/engine.py:742-748`。同理，统计累加必须在 capture 前打开，否则那组 op 不会被捕获进去，`engine/engine.py:700-702`。

**图内可捕获的路径：**

- **decode GPU offload**，`_decode_routed` 的 GPU 分支（`layers/moe.py:285-296`）。文档字符串明说"All device-side with fixed shapes, so the decode call is CUDA-graph capturable"，`layers/moe.py:269-270`。
- **`ensure_experts`**，因为委托给 flashlib 的 `lru_ensure`，固定形状、设备侧（`offload_kernels.py:19-40`）。
- **`copy_missing` 的融合分支**，地址表是常量（`offload_cache.py:385-386`），`layer_id` 是每个图节点的静态 int（`offload_cache.py:1033`）。
- **hybrid decode 整体**，`_decode_hybrid` 文档字符串明说"Capture-safe: the routing split is device-side elementwise and the CPU submit/sync are host nodes."，`layers/moe.py:309-312`。
- **hybrid 的统计累加**，`record_decode_stats_hybrid` 全是设备侧算子（`offload_cache.py:911-926`），并且 `collect_stats` 必须在 capture 前设好（`engine/engine.py:700-702`）。
- **CPU 执行器的 submit/sync**，两条路都是图节点：host-func 路径是 `cudaLaunchHostFunc`（`cpu_moe_ext.cpp:1970-1977`）；flag 路径是 stream memop，注释解释为什么 replay 安全："the WAIT immediate is constant, so CUDA-graph replays are safe"，`cpu_executor.py:266-268`，同样的说明在 `cpu_moe_ext.cpp:63-64`。
- **`moe_offload_cache.reset()` 不能在图内**，所以在 `_capture_graphs` 里捕获前后各调一次（`graph.py:164`、`:195`、`:200`）。

**刻意不在图内的路径，以及为什么：**

- **prefill 全部**。没有捕获 prefill（`graph.py:204-205`），所以 prefill 里的同步拷贝无害。
- **`copy_missing` 的 unpinned 整层同页分支**。注释直接把理由写在代码里："never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)"，`offload_cache.py:1022-1023`。
- **decode 路由直方图**。`collect_decode_freq` 是主机侧 scatter，注释说明为什么在开图时无效："Only accurate with CUDA graphs disabled (the captured graph would not re-run this host-side scatter on replay)."，`offload_cache.py:244-248`。
- **prefill hit-D2D 的主机侧分类**。分类是主机上的 numpy 运算（`offload_cache.py:770-773`），本来就在图外。
- **`BlockWeightStreamer`**。它跑在视觉编码阶段，也就是 prefill 期，`graph.py:204-205` 保证不捕获。`[my inference]`，我核对过 vision 的调用点都在 `encode` / `forward` 里，而 `place_weights` 的 host 模式只在 `config.active_encoders` 时启用（`engine/engine.py:336-344`）。

**统计的图安全性被单独论证过。** `collect_stats` 的注释里写了唯一的图副作用和它的量级："The only graph artifact is a one-off warm-up increment at capture time (<0.1% over a session)."，`offload_cache.py:219-224`。`[read from source]`

---

## 6. 影响"搬运/重叠"的环境开关清单

我先说清检索范围。我 grep 了 `os.getenv` / `os.environ.get` / `os.environ[` 在这几个文件/目录里的全部出现，再加上 C++ 侧的 `getenv`，只列**会改变搬运或重叠行为**的。

| 环境变量 | 默认 | 读取位置 | 效果 |
|---|---|---|---|
| `FREETOKEN_FUSED_COPY` | `"1"`（开） | `moe/offload_cache.py:15` | 置 `0`/`false`/`no`/`off` 关闭多 bank 融合拷贝，`copy_missing` 退回逐 bank 循环。注释说这是保留给 A/B 剖析的（`:12-14`）。另外对齐不满足时也会自动回退（`:406-419`）。 |
| `FREETOKEN_HYBRID_FETCH` | `"recency"` | `moe/offload_kernels.py:15` | 设成 `"lowest_id"` 时，封顶 fetch 改为取最小专家 id（路由无关的原始启发式），默认按 recency（专家级 LRU）取。 |
| `FREETOKEN_CPU_MOE_FLAG_SYNC` | `"1"`（开） | `moe/cpu_executor.py:51` | 置 `0` 完全不用 stream memop 握手，退回每层两次 `cudaLaunchHostFunc`。watchdog 的报错信息里也点名这个开关（`cpu_executor.py:658`）。设备不是 cuda 时也会自动关（`:206`）。 |
| `FREETOKEN_HYBRID_OVERLAP` | `"1"`（开） | `layers/moe.py:26` | 置 `0` 强制串行路径：在 PCIe 抓取加 GPU GEMM 之前先 sync CPU 池（`layers/moe.py:327-329`、`:344`）。注释明说这是"measurement-only escape hatch to A/B the overlap benefit"（`:23-25`）。 |
| `FREETOKEN_SKIP_FAST_INDEX_COPY` | 未设（关） | `kernel/fast_index_copy.py:17`、判定在 `:21-22` | 置 `1`/`true`/`yes`/`on` 后，两个 index-copy 入口都直接 return（`:111-112`、`:180-181`）。注释说明这是"Debug/perf ablation: keep miss bookkeeping intact, but make the copy free to approximate a zero-copy-miss runtime"（`:108-110`）。副作用：prefill hit-D2D 会被判不可用（`offload_cache.py:714-715`）。 |
| `FREETOKEN_BANK_CUDA_ALLOC` | 未设，即走 `born_pinned_default() == False` | `moe/host_banks.py:62`、默认逻辑 `:68-75` | 真值时 PINNED 层用 `cudaHostAlloc`（born pinned+mapped）而不是"懒 mmap + 填完 register"。默认关，理由："registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill"（`:71`）。有 unpinned 层计划时该开关被否决，因为 cudaHostAlloc 会花掉计划本身要省的 pin 配额（`:91-94`）。 |
| `FREETOKEN_SKIP_BANK_PIN` | 未设（关） | `moe/host_banks.py:136` | 真值时 `HostBank.pin()` 变 no-op。文档字符串警告"never set it when serving, the GPU paths need registered banks"（`:133`）。只给 FTW 转换器这类纯 CPU 工具用。 |
| `FREETOKEN_BENCHBW_PATH` | 未设 | `moe/bench_profile.py:88` | 显式指定 profile JSON 路径，优先于按 GPU UUID 找文件。影响 hybrid 判定和 fetch fraction 的取数来源（`:88-101`）。 |
| `XDG_CACHE_HOME` | 未设则 `~/.cache` | `moe/bench_profile.py:35` | 决定 profile 目录 `$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json`（`:39-46`）。 |
| `FREETOKEN_BENCH_PROGRESS` | 未设（关） | `moe/benchbw.py:726` | 置 `"1"` 时向 stdout 打 `FTBENCH done total label` 进度行（`:731-733`），不改变测量本身。daemon 启动 bench 时会设它（`daemon/app.py:349`）。 |
| `FREETOKEN_CPU_MOE_ISA` | 未设（自动取最高支持档） | Python 侧 `moe/benchbw.py:85-97`（`_forced_isa` 上下文管理器）；C++ 侧 `csrc/cpu_moe/cpu_moe_ext.cpp:688` | 把 CPU MoE 的 ISA 档位上限往下压（`scalar` / `avx2` / `avx512` / `avx512bf16`），只降不升。benchbw 用它做 `isa_sweep` 诊断，正式结论（`bw_gbs`）永远用自动选的档（`benchbw.py:509-514`）。 |

**C++ 侧的开关（在 CPU 算专家的路径上，影响时间）**

| 环境变量 | 默认 | 位置 | 效果 |
|---|---|---|---|
| `FREETOKEN_CPU_MOE_SCALAR` | 未设 | `cpu_moe_ext.cpp:681` | 只要设了就强制 scalar 档（legacy 别名）。 |
| `FREETOKEN_CPU_MOE_NO_VNNI` | 未设 | `cpu_moe_ext.cpp:729` | 退出 AVX-VNNI 的 W4A8 路径，用于 A/B。 |
| `FREETOKEN_CPU_MOE_NO_AVX512VNNI` | 未设 | `cpu_moe_ext.cpp:744` | 强制走 256 位路径而不是 512 位，用于 A/B 两个 W4A8 kernel。 |
| `FREETOKEN_CPU_MOE_PF_BLOCKS` | 未设（用内置默认 `min(512 blocks = 4 KB, 2 rows)`） | `cpu_moe_ext.cpp:468`，注释 `:457-467` | 软件预取距离（单位是 16-K 块）。显式设的值**逐字采纳、不做夹紧**，`0` 表示关。注释说明理由："the per-machine optimum can sit past the safe default (+20% at 4 KB on a 24-thread Ice Lake with 256B rows)"。 |

**engine 级、间接触发搬运策略变化的开关**

| 环境变量 | 默认 | 位置 | 效果 |
|---|---|---|---|
| `FREETOKEN_PIN_BUDGET_GB` | 未设；仅在 WSL 上自动算出"物理 RAM 的 40%" | `engine/engine.py:1285`，逻辑 `:1281-1290` | 直接决定 pin 预算。预算存在时会启用 split residency（锁一部分层、pin 一部分层），而 split residency **强制关掉 prefill overlap**（`engine/engine.py:613-621`）。所以在 WSL 上这个变量的存在本身就改变了重叠行为。 |
| `FREETOKEN_UNIFIED_MEMORY` | 未设（走 `cudaDevAttrIntegrated` 探测） | `engine/engine.py:1093`，逻辑 `:1083-1101` | 强制声明/否认这是统一内存 GPU。统一内存 GPU 上 auto 会选 `fused` 而不是 offload（`engine/engine.py:1576-1581`），因为注释说"host banks and the GPU slot cache are the same DRAM, so the offload family's pinned staging + slot gather are DRAM-to-DRAM copies with no PCIe link to hide behind"（`:1085-1087`）。也就是说这个开关能把整条重叠路径关掉。 |

**明确不是重叠开关（免得误记）**

- `FREETOKEN_MOE_CONFIG_DIR`（`moe/fused.py:154`）只是 Triton 调参配置的搜索目录，不改变搬运。
- `FREETOKEN_KERNEL_CACHE_DIR` / `FREETOKEN_DISABLE_KERNEL_CACHE` / `FREETOKEN_DISABLE_JIT`（`kernel/utils.py:11-14`）只影响 kernel 从哪来，不影响算法。
- `weight_stream.py` 和 `kernel/pinned.py` 里**一个 `os.getenv` 都没有**，我 grep 过，无输出。

---

## 7. 我无法从源码确定的事情

这一节是硬的。凡是这里列出的，我没有用"听起来合理"的东西补上。

**关于机制一（`BlockWeightStreamer`）**

1. 我没有找到任何说明"为什么 `_ALIGN` 取 256"的注释或推导。`:12` 只有一个常量。`[无法确定]`
2. `blocks()` 的跨 forward 保护只靠 `begin_event` 加 `release_events`。我没有在任何地方读到对"消费者在别的 CUDA 流上跑 block 计算"这种情况的处理说明。代码隐含假设消费者在本流上排算子。这个假设是否被别的调用点违反，我无法从这五个 vision 文件的源码确定。`[无法确定]`
3. `unstream()` 与正在迭代的 `blocks()` 生成器同时发生的语义没有文档。`unstream` 直接把属性换成设备副本（`:81-86`），而生成器迭代到下一轮会再把属性换成 staging 视图（`:114`）。两者交错会怎样，源码没写。`[无法确定]`
4. 那个"about 60 MiB"只在 `server/args.py:495` 的 help 字符串里出现，注释没有给它是怎么算出来的（哪个模型、多少 block、行宽多少）。`[无法确定]`

**关于机制二（MoE prefill 双缓冲）**

5. `_prefetch_split` 里 `cudaMemcpyBatchAsync` 用的是 `torch.cuda.current_stream(self.device).cuda_stream`（`offload_cache.py:814`）。这个调用点在 `with torch.cuda.stream(self.prefill_copy_stream)` 块内部（`:788`），所以按 torch 的语义它取到的应该是 copy 流本身。我判不出作者是想显式拿到 compute 流的句柄（那样就与 `with` 块矛盾）还是只是顺手用 `current_stream`。两种情况在这个位置的结果相同，但我无法从源码确证作者的意图。`[无法确定作者意图]`
6. hit-D2D 的收益有多大，源码没给数字。我知道的功能事实只有：`moe_prefill_hit_d2d` 默认是 `False`（`engine/config.py:49`），`_hit_d2d_usable` 有五个失效条件。`:703-733` 里没有量化。
7. `_prefill_snapshot_np` 的等价性论证（`offload_cache.py:756-760`）说"Live-vs-snapshot cannot disagree: the only chunk-internal writer (buffer invalidation) rewrites slots already below the 2E threshold"。这是源码里的断言，我没有独立验证它的完备性（也就是"是否真的不存在第二个 chunk 内部的写者"）。`[未验证]`
8. `cache_size` 必须 `> 2E` 才有 hit 区，这只在 `_hit_d2d_usable` 的报错字符串里被当成事实（`:718-722`），没有给"多一个 slot 就能装多少 hit"的量化。`[无法确定]`

**关于机制三（hybrid 协同）**

9. 重叠实际能省多少时间，源码只给了一个不同场景的数字（`cpu_executor.py:308-309` 的 ds_fp4 prequant：`12.85 -> 15.65 tok/s`）。**那不是重叠的收益**，而是把 FP8 round-trip 从 CPU 挪到 GPU 的收益。重叠本身在 `layers/moe.py` 和 `cpu_executor.py` 里没有任何数字。`[无法确定]`
10. `cpu_executor.py:34-36` 提到"2 calls per MoE layer per decode step (~6 ms/step on a 75-layer model)"。这是 host-func 路径的代价估计，没有说是实测还是推算。`[无法确定]`
11. 我没有实机跑过，所以无法确认 flag 握手在真实驱动上是否真的走通；代码里 `memops_probe` 是运行期功能探测（`cpu_moe_ext.cpp:619-628`），静态读代码判不出结果。`[未验证]`
12. `_FLAG_SLOTS_PER_LAYER = 16` 的理由是"more than that is unheard of"（`cpu_executor.py:54`），这是断言式的，源码没有给"实际上见过几个不同 batch size"的数据。`[无法确定]`

**关于 `q*` 拆分**

13. **`q*` 这个名字跟代码对不上。** 我在整个仓库 grep 过 `q_star`、`qstar`、`q^\*`，源码里没有任何符号叫这个。`[read from source]` 我把它映射到"hybrid/offload 判定阈值 2.0"加"fetch fraction 公式"上，这是**我的推断**，不是仓库的说明。`[my inference]`
14. 那个 `2.0` 阈值是怎么定出来的，源码只说"default 2x"（`benchbw.py:17`、`:951-952`），没有任何推导或实测依据。`[无法确定]`
15. `bench_profile.py:187` 的 `min(1.0, pcie_ov / (pcie_ov + cpu_ov))` 和 `:190` 的 `min(1.0, pcie / cpu)` 这两个公式**形状不同**（一个有加法一个没有）。文档字符串用"full-DRAM-contention assumption"来解释为什么退化形式是 `pcie/cpu`（`:168-170`），但 `min(1.0, ...)` 这个夹紧在两条分支上都会截断，源码没说截断后会不会打破"两侧同时结束"的目标。`[无法确定]`
16. `measure_overlap_bw` 用"每次拷贝都同步"来保证 DMA 真的在飞（`benchbw.py:589`），但同步本身给 PCIe 侧引入了空闲间隙。这个间隙对测出来的 `pcie_ov` 偏低多少，源码没量化。`[无法确定]`
17. 独立测量与竞争测量的具体数字，我一个都没跑。我**不写任何 benchmark 数字**，只引用仓库自己写下的那几处（`offload_cache.py:22`、`:777`、`fast_index_copy.py:166`、`cpu_executor.py:309`、`host_banks.py:9-10`、`cpu_executor.py:35-36`），并且标明是引用。

**关于 `fast_index_copy` 与 pinned**

18. `load_nc` 的 `ld.global.L1::no_allocate` 和 `store_nc` 的 `st.global.wt` 为什么选这两个具体修饰符，源码里**没有解释**。我只能从指令语义说它绕开了 L1 分配和 L2 缓存，为什么这是对的，是我的推断。`[my inference]`
19. "70GB bank 会预留 128GB"这个数字来自 `pinned.py:5-6` 的注释，是举例式的。我没有验证 torch 的 pinned 分配器在这个版本上的确切取整规则。`[未验证]`
20. `_shrink_worker_feature_size`（`fast_index_copy.py:70-75`）在 `feature_size < worker_feature_size` 时下调，然后循环除以 2 直到能整除或降到 128。这个 128 的下界为什么是 128，源码没解释。`[无法确定]`

**关于支撑机制**

21. `flashlib` 的 `lru_ensure` 是外部包（`pyproject.toml:40-41` pin 在 `flashlib==0.3.0`），它的内部实现不在本仓库，所以 `ensure_experts` 的**槽位选择策略**我只能从调用点和 `lru_stats` 的形状反推，看不到内核。`[无法确定]`
22. `MARLIN_MAX_CACHE_SIZE = 992` 的来源在 `offload_cache.py:98-100` 的注释里给了（`moe_align_block_size` 要求 `round_up(experts, 32) < 1024`），但那是 vLLM 的约束，我没去核对 vLLM 侧的代码。`[未验证]`
23. FTW 的 `_BANK_CONCURRENCY = 4`（`ftw.py:57`）和线程池尺寸 `min(max(_BANK_CONCURRENCY, 16), max(n_jobs, 1))`（`:591`）为什么是这两个数，源码没解释。`[无法确定]`
24. FTW 的 O_DIRECT 直读是否真的比 mmap 快，以及快多少，源码只有定性说法（`ftw.py:226-231`），没有数字。`[无法确定]`
25. `_BANK_BYTES_PER_EXPERT`（`offload_cache.py:86-96`）是估算用表，注释要求"keep in sync with _BANK_SCHEMAS"（`:84`），但没有任何机制强制两者一致。我没找到校验它们的测试。`[未验证]` 这张表被 `moe/expert_banks.py:265` 用来估 bank 大小，而估出来的数会进 pin 预算判断（`engine/engine.py:605-612`），所以不一致会走错分支。这是我读出来的**风险点**，不是确定的 bug。`[my inference]`

**其他**

26. 我没有跑任何 GPU 测试，也没有安装任何东西（任务限制）。所以本文全部是静态阅读结论，不含我自己产生的实测数字。
27. `engine/engine.py:1694` 那处 `override("moe_prefill_overlap", True)` 我已经读清了：它在一个 `if is_moe and config.moe_strategy == "cpu":` 块里（`:1685`），同一块里还 `override("moe_cache_size", 2 * num_experts)`（`:1693`）并关掉 `moe_cache_auto`（`:1691-1692`）。也就是 `--moe-strategy cpu` 时，slot cache 被钉死成正好 `2 * num_experts` 个槽，prefill overlap 被强制打开。`[read from source]` 一个顺带的结论：因为此时 `cache_size == 2 * E`，而 hit-D2D 要求 `cache_size > 2 * E`（`offload_cache.py:718-722`），所以 **`--moe-strategy cpu` 下 hit-D2D 永远不可能生效**，即使用户显式传了 `--moe-prefill-hit-d2d`。`[my inference，依据是那两个条件的直接比较]`
