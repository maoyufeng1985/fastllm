# 两个长请求能否轮流用 prefill chunk：复核与结论（2026-09-15）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP。
目标：判定"两个并发长请求轮流使用 prefill chunk"（时间片轮转）是否成立，
以及它对 TTFT 与吞吐的真实影响。全部结论来自实测，不靠推断。

## 0. 结论先行

1. **轮转不成立，而且是刻意关闭的**，不是没实现。`src/models/basellm.cpp:1754`
   的注释写明：in-flight chunked prefill 排在新 prompt 之前，目的就是"不让两条
   80K 请求来回 ping-pong 2048 的块"。
2. **串行已被两组独立数据确认**：`inFlight` 在任何一轮都不超过 1；2×20K 的
   TTFT 是 8.2 s 对 16.5 s（正好一倍）。
3. **轮转实现了，它暴露了一个真 bug，然后这个 bug 修好了。** 轮转让
   `inFlight` 第一次达到 **2**——两条长 prefill 真正同时在飞——而那一刻**段错误**。
   根因是主机侧空指针（分叉批量前向对空返回取下标 0），不是 GPU 故障；已修
   （提交 `56e3445f`）。修后 2×200K × FP8 真正并发：`inFlight=2` 97 轮、
   TTFT 253.3/254.6 s（§3.3、§3.4）。
4. **但它的收益只有公平性。** Total time 与串行相同（约 255 s），#0 的 TTFT
   从 127.4 s 变慢到 253.3 s。所以缺省关闭，只在并发长请求是常态时开。
4. **但轮转买不到吞吐。** 两条长请求的 prefill 总计算量不变，轮转只改变谁先
   完成：后者 TTFT 显著改善，**前者 TTFT 等比例变差**。净效果是延迟的再分配。

## 1. 轮转为什么被关掉（机制，代码位）

```cpp
// src/models/basellm.cpp:1754-1757
// In-flight chunked prefills stay ahead of new prompts so
// two 80K requests do not ping-pong 2048-token slices.
orders.push_back({PrefillOrderSortKey(it.second), it.first, it.second});
```

`PrefillOrderSortKey`（`include/models/longPrefillChunk.h`）给**已经喂过 token**
的 prefill 一个巨大负偏置：

```cpp
if (ctx->prefillRemaining > 0 && ctx->preTokens > 0) {
    return -(remaining + 1000000000);
}
```

于是它在整个排序里永远第一。配合准入预算（`GetBatchedPrefillTokenLimit()`
在用户显式设 `chunked_prefill_size` 时直接返回它），同一轮里第一条占满预算、
第二条算出 0 长度 chunk 被跳过。结果：第一条一路喂到跑完，第二条才交棒。

## 2. 实测：串行，两组独立数据

### 2.1 `inFlight` 从不超过 1

给调度日志加了请求身份字段（`sel` / `preTok` / `rem` / `inFlight`，
`qwen3_5.cpp` 的 `FASTLLM_SCHED_TRACE` 分支）。2×20K、缺省预算：

| 指标 | 实测 |
|---|---|
| `inFlight` 分布 | `0`：34 轮，`1`：29 轮，**`2`：0 轮** |
| `selected=2` 次数 | 21 次（都发生在两条都还是 `preTokens=0` 时）|
| TTFT min / max | **8199 ms / 16495 ms**（正好一倍）|
| Total time | 16.90 s |

`selected=2` 出现 21 次却从没让 `inFlight` 到 2，说明"两条同轮入选"只发生在
两者都还没开始喂 token 的时候；一旦有一条领先，它就一直领先到跑完。

### 2.2 200K × 2 的 TTFT 也是一倍关系

FP8 KV、`--tokens 500000`、`--batch 2`、`--input_tokens 200000`：

| 指标 | 实测 |
|---|---|
| TTFT min / max | **127.8 s / 255.8 s** |
| 并发窗口 | 只有 #1 在跑（36–39 tok/s），#0 已退场 |
| `prefillBlocked` | 0 次（页子够，不是页的问题）|

两级长度（20K 与 200K）都给出"max ≈ 2 × min"，这是串行的独立佐证。

## 3. 崩溃：更正与真复现

### 3.1 更正上一版的两条错误复现

本文件初版把以下两跑记成"段错误"，并据此断言"同批两条 prefill 必崩"：

| 配置 | 初版记录 | **干净源码复跑** |
|---|---|---|
| chunk 1024 / 预算 2112 | Xid 31 + 段错误 | **exit 0，不崩** |
| chunk 1024 / 预算 2048（2×8K）| 段错误 | **exit 0，不崩** |

两次复跑都不崩。当时的库处于**撤改代码后重建的中间状态**，所以那两条是脏构建
的假象。**更正：不能用它们支撑"同批两条必崩"的结论。**

### 3.2 根因：分叉批量前向对空返回取 [0]（已修复，提交 `56e3445f`）

崩溃不是 GPU 故障，是主机侧空指针读取，落在 `runSplitBatchForward`：

```cpp
std::vector<int> curRet = ForwardGPU(1, ...);
ret.push_back(curRet[0]);   // 空向量的 _M_start 为 0 -> 读地址 0
```

两处同样写法：`ForwardGPUWithHiddenStates`（原 15348）与 `ForwardV2` 副本
（原 32584）。

机制链条，每环都在代码里核过：

1. 调度器在 `skipIntermediateHead` 为真时经 `IntermediatePrefillGuard` 置位
   `isIntermediateChunkedPrefill`（`qwen3_5.cpp:23068-23090`）。该条件要求
   **全部被选中的请求都贪心、且 `prefillRemaining > seqLens[i]`**（23073）——
   也就是两条都在分块 prefill 中途。
2. `ForwardSingleGPU`（15126）据此 `logits.FreeSpace()` 并 `return {}`。
   **中间分块本来就不该产出 token，空返回是既有协议**：batch-1 路径与
   `ForwardV2`（32592 直接 `return chunkRet`）都这么传播。
3. `canRunFusedBatchPrefill` 对**已分配 GDN 状态**的线性层返回假
   （15408-15410，注释写明 "continued or chunked request keeps the
   request-local path"），于是 `ForwardGPUWithHiddenStates` 走
   `return runSplitBatchForward()`（15425）。
4. 分叉路径逐条调 `ForwardGPU(1, ...)`，拿到空向量后仍取 `[0]`，崩。

**修法是跳过而不是兜零**：兜零会往调度器塞一个假 token，而这一轮本就不该有
token。两处都加 `if (curRet.empty()) continue;`。

**这条更正了本文件初版的一个论断**：初版说"同批两条 prefill 是一条从未走过的
路径"。不成立——分叉路径本身是常规路径，批量 decode 也走它；没被走过的是
"**两条同时在分块中途 + 分叉路径 + 跳过 head**"这个组合。而缺省的准入预算
（只放一条 chunk）恰好让这个组合不可达，所以串行一直在掩盖它。

**为什么热启动没覆盖**：`ragged batched serving` 预热确实跑过一个 4 条序列的
ragged eager prefill，但它用自己新建的连续 KV `Data`，不碰分页管理器，也从不
设置 `isIntermediateChunkedPrefill`，所以到不了那个 `return {}`。它覆盖了形状，
没覆盖状态。

### 3.3 修复后的实测（2×20K，轮转开）

| 指标 | 串行缺省 | 轮转（修复前） | 轮转（修复后） |
|---|---|---|---|
| exit | 0 | **139 段错误** | **0** |
| `inFlight=2` 轮数 | 0 | 2 | **10** |
| TTFT min / max | 8199 / 16495 ms | — | **16358 / 16361 ms** |
| `before last TTFT` | 10 / 0 | — | **0 / 0** |
| Total time | 16.90 s | — | 16.77 s |
| 逐请求窗口速率 | 74.73 / 77.01 | — | 75.17 / 75.11 |

**轮转前后 sha 完全相同（`3c051feffab1db8c`）**，与改动前也相同——轮转是纯调度
改动，不换数值。缺省（轮转关）回归：exit 0、Total time 16.8850 s、`inFlight`
上限 1，与改动前一致。

### 3.4 目标组合：2×200K × FP8 现在真正并发

```
FASTLLM_PREFILL_ROTATE=1
--tokens 409600 --gpu_mem_ratio 0.98 --kv_cache_dtype fp8_e4m3
--input_tokens 200000 --output_tokens 8 --batch 2 --max_batch 2
```

**exit 0、阻塞 0、无 OOM**，`inFlight=2` 出现 **97 轮**，TTFT **253.3 s /
254.6 s**（两条同时出首 token），窗口内逐请求 26.24 / 26.14 tok/s。

对照串行：TTFT 127.4 s / 254.8 s，Total time 约 255 s。**两者 Total time 相同，
轮转买到的是公平性，#0 的 TTFT 变慢一倍。** 所以缺省必须保持关闭：单条长请求
是常态时轮转纯亏。

### 3.5 真复现（保留作回归样本）：两条 in-flight prefill 会崩

干净源码、`FASTLLM_PREFILL_ROTATE=1`（轮转把预算抬到 2×chunk，让两条同时
in-flight）后，最小样本段错误：

```
FASTLLM_PREFILL_ROTATE=1 FASTLLM_SCHED_TRACE=1 python3 -m ftllm.cli benchmark \
  /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding --max_batch 4 \
  --tokens 65536 --dtype auto --enable_thinking false --prefix_cache false \
  --input_tokens 20480 --output_tokens 32 --batch 2 --warmup 0 --temperature 0 --top_k 1
```

结果：**exit 139（段错误）**，而调度日志显示 `inFlight` 达到 **2**——这是本项目
里第一次让两条长 prefill 真正同时在飞。此前所有配置（缺省、chunk 1024/2048、
预算 2048/2112/4096）的 `inFlight` 上限都是 1，串行**掩盖**了这个 bug。

所以真实情况是：

- 串行不是"实现不了并发"，而是**并发的 bug 被串行挡住了**；
- 一旦让两条同时 in-flight（轮转或放预算），就踩到它；
- 这也解释了 §3.1 的假象：当时我以为改预算就能复现，其实那两次是脏构建。

## 3.6 服务模式验收（2026-09-15，真实到达模式）

benchmark 的 `--batch 2` 是**同时**提交两条，TTFT 243–255 s 对两条都成立。真实
服务里请求是陆续到的，形状不同，所以单独验了一次。

### 配置

```sh
FASTLLM_PREFILL_ROTATE=1 python3 -m ftllm.cli serve \
  /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding --max_batch 2 \
  --tokens 409600 --gpu_mem_ratio 0.97 --kv_cache_dtype fp8_e4m3 --dtype auto \
  --enable_thinking false --prefix_cache false --port 18123 --host 127.0.0.1
```

两条 200K 请求错开 5 s 发出（`/v1/chat/completions`）。

### 结果：第一条不再吃满 2x 惩罚

| 请求 | 发出 | 耗时 | 状态 |
|---|---:|---:|---|
| A | t+0 s | **247.8 s** | ok |
| B | t+5 s | **243.0 s** | ok |

服务端 trace：`inFlight=2` **90 轮**，且前 6 轮是 `orders=1 / inFlight=1`——
A 独占了相当于到达间隔的时间，B 到达后才开始交替：

```
it=0..5  orders=1 selected=1 sel=0 inFlight=1   ← A 独跑
it=6     orders=2 selected=2 sel=0 inFlight=2   ← B 到达，两条并排
it=7..   orders=2 selected=2 sel 在 0/1 交替
```

**这一点纠正了 benchmark 给人的印象**：同时提交时两条都等 ~253 s，而错开到达时
A 只等 247.8 s、B 243.0 s——**接近"各付一半"，不是"第一条翻倍"**。轮转让后到
的请求快速追上，同时先到的请求保留它抢跑的那一点。

### 显存：峰值由池子决定，不由在飞请求数决定

逐秒采样（1 s 间隔）：

| 场景 | GPU0 峰值 | 余量 |
|---|---:|---:|
| 单条 200K | 15711 MiB | 673 MiB |
| 两条错开到达 | 15711 MiB | 673 MiB |
| 同时提交（benchmark） | 15707 MiB | 677 MiB |

**三者完全相同。** 原因是页池在启动时按 `--tokens` 分配，空闲页不省显存：
`--tokens 409600` → 3200 页 × 1.05 MB = **3.36 GB**，与请求数无关。

所以：

- 这里的 673 MiB 余量**不是"还有空间"**，池子已经按上限落地了；
- **省显存只能靠右调 `--tokens`**，而 2×200K 需要 3126 页、除以 80% 上限
  （`promptLimit = totalPages * 4 / 5`）得池子 ≥3908 页，**409600 已经是下限**，
  没有可回收的余量。要更多余量只能减少每条请求的上下文或换卡。

### 天花板：2 并发是硬的，第三条被排队

三条 200K 错开 4 s 到达：

| 请求 | 耗时 | 状态 |
|---|---:|---|
| R0 | 248.3 s | ok |
| R1 | 248.4 s | ok |
| R2 | **364.6 s** | ok（被排到后面） |

R2 多等约 116 s，说明它没有和前两条重叠——池子装不下第三条的 1563 页。
服务没崩、请求没失败，是页池不足的正确降级。**所以 200K 档的并发上限就是 2。**

## 4. 轮转的收益：只有公平性，没有吞吐

两条 200K 的 prefill 吃同一份算力，总计算量不变，所以：

| | 现在（串行） | 时间片轮转 |
|---|---|---|
| #0 的 TTFT | **127.8 s** | ~256 s（退化） |
| #1 的 TTFT | 255.8 s | ~256 s（持平） |
| 两条都出首 token | 255.8 s | ~256 s |
| Total time | 256 s | 256 s |

**轮转把 #0 的 TTFT 从 128 s 推到 256 s，换 #1 不被饿死。** 这是一个延迟
再分配，不是提速。它值不值得，取决于"并发长请求是否常见"以及"公平性是否比
单条延迟更重要"：

- 若并发长请求是常态，轮转有意义（避免后面的请求被无限期挂住）；
- 若单条长请求是常态，轮转是纯亏（每条都变慢）。

**技术上的关键区别**：轮转是每轮只跑一条 chunk（batch-1 前向），**绕开了 §3 的
崩溃路径**；而"同批两条"必须修掉那个段错误才能用。所以如果要在当前代码上拿
并发 prefill，轮转是唯一不需要先修 bug 的形状。

## 5. 复现方式

```sh
# 串行证据（inFlight 从不超过 1；TTFT 1:2）
FASTLLM_SCHED_TRACE=1 python3 -m ftllm.cli benchmark \
  /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding \
  --max_batch 4 --tokens 65536 --dtype auto --enable_thinking false \
  --prefix_cache false --input_tokens 20480 --output_tokens 32 --batch 2 \
  --warmup 0 --temperature 0 --top_k 1

# 崩溃复现（2×8K，最小可复现样本）
FASTLLM_SCHED_TRACE=1 FASTLLM_PREFILL_BATCH_TOKEN_LIMIT=2048 python3 -m ftllm.cli benchmark \
  /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding \
  --max_batch 4 --tokens 32768 --dtype auto --enable_thinking false \
  --prefix_cache false --input_tokens 8192 --output_tokens 32 --batch 2 \
  --chunked_prefill_size 1024 --warmup 0 --temperature 0 --top_k 1
```

注意：`FASTLLM_PREFILL_BATCH_TOKEN_LIMIT` 是**实验用**开关，已在验证后从代码里
撤除，所以崩溃复现需要临时加回（改动只有 `GetBatchedPrefillTokenLimit()` 一个
提前返回，见 commit 说明）。撤除的理由是它唯一的作用是通往一条会崩的路径。

## 6. 代码处置

- `src/models/qwen3_5.cpp` 的调度诊断字段（`sel` / `preTok` / `rem` /
  `inFlight`）**保留**：零开销、`FASTLLM_SCHED_TRACE` 缺省关，是回答"到底谁在
  跑"的唯一手段。
- **轮转实现保留，缺省关**：`FASTLLM_PREFILL_ROTATE=1` 时
  `PrefillOrderSortKey` 在 in-flight 分支里改用轮转票（`ResponseContext::
  prefillTicket`，在每次入选 prefill 时盖新票），`GetBatchedPrefillTokenLimit()`
  同时把预算抬到 2×chunk——两者缺一不可：只改排序会在 `alreadyInBatch=1` 时
  因 `SelectPrefillChunkLen` 返回 0 被跳过（实测 trace 里 `sched=0`），排序根本
  走不到。
- **缺省行为逐位不变**：轮转关时 2×20K 复跑 exit 0、Total time 16.90 s、
  `inFlight` 上限 1，与改动前一致（改动前是 16.8993–16.90 s）。

保留轮转代码的理由不是"它能用"（开着自己会崩），而是**它是暴露并复现
§3.2 那个 bug 的唯一开关**。修 bug 时需要一个能稳定触发 `inFlight=2` 的入口，
删掉它下次还得重新实现。
