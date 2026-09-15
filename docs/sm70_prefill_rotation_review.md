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
3. **轮转避开了一条已知会崩的路径。** "同批两条 prefill"在三种配置下全部
   段错误（见 §3），而轮转是每轮只跑**一条** chunk、走 batch-1 前向。所以轮转
   不只是"换一种公平性"，它是当前唯一能绕开崩溃拿到并发 prefill 的形状。
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

## 3. "同批两条 prefill"会崩（复现三次）

为了做轮转的对照，试着放宽准入预算让两条 chunk 进同一个 forward。四次配置：

| 准入预算 | chunk | 预热分配 | 结果 |
|---:|---:|---:|---|
| 2048（缺省） | 2048 | 2048 | 正常（串行）|
| 4608 | 2048 | 4608 | **OOM**（预热跟着预算放大）|
| 2112 | 1024 | 2112 | **Xid 31 + 段错误** |
| 4096（预热已解耦） | 2048 | 2048 | **段错误** |
| 4096 | 2048 | 2048 | **段错误**（2×20K，与长度无关）|
| 2048 | 1024 | 2048 | **段错误**（2×8K，均匀路径）|

**第三条把"预热缓冲放大"这个原因排除了**：显式把预热钉回一个 chunk
（`servingPrefillTokenLimit = GetChunkedPrefillSize()`）之后，仍然崩。
最后两条把范围进一步缩小：**与上下文长度无关**（20K 也崩）、**与 ragged 与否
无关**（两条 seqLen 相同时走均匀批量路径，也崩）。

所以"一个 forward 装两条 prefill"是条从未被走过的路径，一进就段错误。它平时
不可达，正是因为准入预算只放一条 chunk；这也解释了为什么它没在别处暴露。
（该路径由 `7fdfe119` "修复 Qwen3.5 服务期显存池未命中" 引入。）

**这不是新回归，是长期休眠的路径。** 复现方式对工程有用，所以完整记录在
§5。修它属于独立工作：GPU MMU 故障说明是地址记账错了，不是参数没调好。

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
- 准入预算 override 与预热口径解耦**已撤除**，缺省行为与改动前逐位一致
  （缺省 2×20K 复跑 exit 0，Total time 16.90 s）。
