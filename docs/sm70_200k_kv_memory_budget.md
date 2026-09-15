# 200K 上下文的每卡 KV 显存账：省显存路线（2026-09-15）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP。
目标：找出 200K 上下文下每卡 KV 显存的可动项，并判定哪些真能腾出空间。

## 0. 结论先行

1. **每页 KV 是唯一能成倍改的项**：FP16 是 2.10 MB/页，FP8 是 1.05 MB。其余
   都是线性或固定的。
2. **页池不是"超配"，是算出来的**（`fitPagesWithLinearReserve`，二分搜最大
   可行页数，逐项扣每页字节、延迟页缓存、每请求线性缓存、运行时预留，多卡取
   最小值）。所以不存在"池子白占"这回事。
3. **`--gpu_mem_ratio` 是现成的、有效的大杠杆**：0.90 → 0.95 释放
   **846 MB/卡**，0.90 → 0.98 释放 **1.36 GB/卡**。但它换的是运行时余量，
   不是白给。
4. **2×200K 的硬缺口是 GB 级，不是这个量级**（见 §4），单靠调 ratio 补不上。

## 1. 页池是算出来的，不是配额

`src/models/basellm.cpp:4817-4871` 的 `fitPagesWithLinearReserve` 对每张卡做
二分搜索，求"满足以下全部约束的最大页数"：

```
页数 × 每页 KV 字节
+ 页数 × 延迟页缓存/页
+ activeBatch × 每请求线性缓存
+ runtimeReserve(activeBatch)
<= avail
```

被压时会打印 `limit pages X -> Y`。多卡取各卡最小值。所以池子大小是**内存的
函数**，`--tokens` 只是它的上限。

实测佐证（缺省 `gpu_mem_ratio=0.90`）：

| 跑 | 请求的 `--tokens` | 实得池子 | 每页 | 池占 |
|---|---:|---:|---:|---:|
| 8K 档 | 32768 | 256 页 | 2.10 MB | 0.54 GB |
| 单条 200K（FP16）| 204800 | 1600 页 | 2.10 MB | 3.36 GB |
| 双条 200K（FP8）| 500000 | 3907 页 | 1.05 MB | 4.10 GB |

**注意 `availForKV` 不是分配上限**：它在 8K/32768/49152 三种配置下都报
1.93 GB，而 FP8 那次实际拿到 4.10 GB。别拿 `availForKV` 当预算，它只是预热期
的一个中间量。

## 2. `--gpu_mem_ratio`：846 MB/卡 的现成杠杆

`gpuMemRatio` 缺省 0.9（`src/fastllm.cpp:296`），语义是"预留总显存的
`(1 - ratio)` 不动"。CLI 暴露为 `--gpu_mem_ratio`（`tools/fastllm_pytools/llm.py`
的 `set_gpu_mem_ratio`）。

实测（8K、`--tokens 32768`，池子远小于上限，所以只反映预留的变化）：

| ratio | `reserved` | `availForKV` |
|---:|---:|---:|
| 0.90（缺省）| 1.69 GB | 2.05 GB |
| 0.95 | **0.85 GB** | **2.90 GB** |
| 0.98 | **0.34 GB** | **3.41 GB** |

**每降 0.05 释放约 846 MB/卡。**

代价与边界：`reserved` 之外还有两道与它无关的保留——`runtimeHeadroom`
（`min(max(512 MB, 总显存 1%), 2 GB, 空闲/4)`，实测 536.87 MB）与
`servingReserve`（实测 201.33 MB）。它们由 `getCudaRuntimeHeadroom` 与
`servingReserve` 独立计算，**不随 ratio 变化**。所以调高 ratio 不是把所有余量
抽干，而是把"额外那 10%"还给 KV。已知最紧的场景是 **80K C=1 的 warmup/capture
瞬时只剩 271 MiB**（稳态余 1435 MiB），那是 0.90 下的数字；调高 ratio 后这个
余量会被池子吃掉，长上下文工况需要重新量峰值。

## 3. 固定项（省不掉）

| 项 | 大小 | 说明 |
|---|---:|---|
| 模型权重分片 | **5.14 GB/卡** | TP4 后的下限，省不掉 |
| CUDA graph 共享工作区 | 1.69–1.87 MB + KV 记账预留 128–192 MB | 已很小 |
| sampling buffers | 68.6–73.07 MB | 随 vocab × batch |
| 预热激活缓冲 | 随 chunk 切片 | 不是每请求线性缓存 |

## 4. 2×200K 的账：缺口是 GB 级

需求：`2 × 200000 / 128 = 3126 页`。

| KV 精度 | 每页 | 页池需求 | 相对 0.90 的 `availForKV` 2.05 GB |
|---|---:|---:|---|
| FP16 | 2.10 MB | **6.56 GB** | 差 4.5 GB |
| FP8 | 1.05 MB | **3.28 GB** | 差 1.2 GB |
| FP4 | 0.59 MB | 1.84 GB | 够 |

所以：

- **FP16 不可能**（差 4.5 GB，任何 ratio 都补不上）。
- **FP8 差约 1.2 GB**：`--gpu_mem_ratio 0.95` 能补 0.85 GB，0.98 能补
  1.36 GB。但补上之后运行时余量同时被抽掉，而 200K 的 prefill 峰谷本来就紧
  （单条 200K FP16 的池是 3.36 GB，在 0.90 下就只剩一点点空间）。**能不能用
  必须实测峰值，不能用静态账判。**
- **FP4 静态账够**，但实测代价已知：KV 量化在越长上下文越亏（80K fp4 −4.9%、
  180K fp4 −8.2%、fp8 −11.0%），而且 200K 上没量过；它改变数值路径，需要
  质量门。

## 5. 还没做的事（按优先级）

1. **量 `--gpu_mem_ratio` 对长上下文峰值的影响**：0.90 下 80K C=1 的
   warmup/capture 瞬时只剩 271 MiB，调到 0.95/0.98 后池子变大、余量变小，
   必须逐秒采样确认不 OOM。这是"能不能靠 ratio 补 1.2 GB"的直接前提。
2. **`--low_gpu_mem`（CLI 已暴露）**：官方说法是"强制关闭 CUDA embedding 与
   GPU token handoff，保留原有 CUDA…"，我还没量过它省多少、代价多少。
   CUDA embedding 关掉会走 CPU 侧，可能换来可观显存但掉速度。
3. **`--kv_cache_limit`（CLI 已暴露）**：与 `--tokens` 的关系没查清。
4. **每请求线性缓存与延迟页缓存的实际占比**：它们已在预算里扣除，但没单独量过
   数值，也不知道能否压。
5. **FP4 KV 在 200K 的质量门与速度**：静态账是唯一够的选择，但代价未量。

## 6. 复现方式

```sh
# ratio 对预留与 availForKV 的影响（几十秒一跑）
for r in 0.90 0.95 0.98; do
  python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
    --tp 4 --cuda_embedding --max_batch 2 --tokens 32768 --gpu_mem_ratio $r \
    --dtype auto --enable_thinking false --prefix_cache false \
    --input_tokens 8192 --output_tokens 8 --batch 1 --warmup 0 \
    --temperature 0 --top_k 1 2>/dev/null | grep -E "reserved|availForKV"
done
```

注意别用 `--tokens 400000` 做这个测量：FP16 下池子就要 6.6 GB，会直接
`cudaErrorMemoryAllocation`，三种 ratio 都 OOM，量不出 ratio 的效果（本轮踩过）。
