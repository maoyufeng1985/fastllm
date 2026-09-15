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
3. **轮转实现了，而且它暴露了一个真 bug，不是避开一个。** 轮转（本仓库新增
   `FASTLLM_PREFILL_ROTATE`）让 `inFlight` 第一次达到 **2**——两条长 prefill
   真正同时在飞——而那一刻**段错误**（§3.2）。所以"两条同时 in-flight"是坏的，
   串行一直在掩盖它。§3.1 还更正了本文件初版的两条假复现。
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

### 3.2 真复现：两条 in-flight prefill 会崩

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
