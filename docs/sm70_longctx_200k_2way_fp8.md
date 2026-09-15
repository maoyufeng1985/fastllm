## 200K 上下文 × 2 并发 × FP8 KV 实测（2026-09-15）

**问题**：这台机器（4×V100-16GB、TP4）能不能跑 200K 上下文、2 条并发、FP8 KV。

**结论：FP8 KV 是必要的，但不够。内存装得下，调度器不让两条 prefill 重叠，
强行让它重叠会踩 GPU。所以跑起来是串行，不是 2 并发。**

### FP8 确实把每页减半

`--kv_cache_dtype fp8_e4m3` 下 `localKVPerPage` 从 **2.10 MB 降到 1.05 MB**
（每卡、16 层全注意力、128 token）。200K 一条 = 1563 页：

| KV 精度 | 每页 | 1×200K | 2×200K | 池子上限（`--tokens 435200`）|
|---|---:|---:|---:|---:|
| FP16 | 2.10 MB | 3.28 GB | 6.56 GB | 2720 页（不够两条）|
| **FP8** | **1.05 MB** | **1.64 GB** | **3.28 GB** | 3125 页（`--tokens 500000`）|

FP16 下 2×200K 要 6.56 GB KV，加模型分片 5.14 GB 就超 16 GB，**必须 FP8**。

### 页池够，但两条仍然串行

`--tokens 500000 --kv_cache_dtype fp8_e4m3 --batch 2 --input_tokens 200000`：

- 页池 3907 页（4.1 GB），80% 上限 3125 页，**2×200K 需要 3126 页**，刚好够。
- `prefillBlocked` 全程 **0 次**，页守卫一次没触发。
- 结果：TTFT min 127.8 s、**max 255.8 s**，窗口里只有 #1 在跑，#0 早退场。
  **完全串行。**

### 为什么串行：批量准入上限 = chunk 尺寸

`Qwen3_5Model::GetBatchedPrefillTokenLimit()`（`src/models/qwen3_5.cpp:10116`）
在用户显式设了 `chunked_prefill_size` 时**直接返回它**，把"每请求切片大小"当成了
"多少 token 可以进同一个 forward"。chunk = 2048 时：

1. 第一条请求进批，`selectedPrefillTokens = 2048`；
2. 第二条请求调 `SelectPrefillChunkLen`，`alreadyInBatch=1` 走到
   `curLen = min(curLen, 2048 - 2048) = 0`，返回 0；
3. 调用方 `if (scheduledTokens <= 0) continue;` 把它跳过。

SCHED_TRACE 佐证：`canAddPrefill=1` 全程为真（页子够），但 187 次迭代里 186 次
`selected=1`，只有 1 次 `selected=2`。**不是页不够，是准入预算只够一条的 chunk。**

### 试着解耦这个上限，撞到两条墙

加了 `FASTLLM_PREFILL_BATCH_TOKEN_LIMIT`（只把准入预算抬高、不改 chunk），三档都失败：

| 准入预算 | chunk | 结果 |
|---:|---:|---|
| 4608 | 2048 | OOM：`cuda malloc failed ... dims=[1, 4608, 8704]`（80 MB）|
| 2112 | 1024 | **Xid 31（GPU MMU 故障）+ 段错误**，崩在 `ragged batched serving` 预热之后 |

第二档的 dmesg：

```
NVRM: Xid (PCI:0000:05:00): 31 ... MMU Fault: ENGINE GRAPHICS GPCCLIENT_T1_0
python3[...]: segfault at 0 ... in libfastllm_tools.so
```

日志顺序说明一切：`serving eager prefill warmup (ragged batched serving): batch 4,
total tokens 2112` → `[sched] it=1 orders=2 selected=2` → 崩。**"两条 chunk 同批"
是一条从未被走过的路径**，一进去就踩 GPU 地址。

两条墙叠在一起：
1. 抬高准入预算会让 warmup 缓冲区同比放大（`servingPrefillTokenLimit` 取
   `GetBatchedPrefillTokenLimit()`，`qwen3_5.cpp:10558-10562`），而卡上只剩
   几 MB 余量；
2. 就算放进去，ragged batched prefill 那条路径自身有 bug（Xid 31）。

### 处置

实验代码（env 开关）**已全部撤除**，工作区与 `src/` 保持干净，二进制已重建。
这两个问题都不该在没摸清 ragged batched prefill 生命周期的情况下继续试——
GPU MMU 故障说明是地址记账错了，不是参数没调好。

### 这台机器上的可选做法

- **200K × 1 并发：可行**（FP8 或 FP16 都能跑，FP16 单条实测 TTFT 127.6 s）。
- **200K × 2 并发：当前不可行**，卡在调度器准入，不在显存。
- 若坚持要 2 并发，需要先修 ragged batched prefill 的地址记账（Xid 31）并把
  warmup 缓冲与准入预算解耦，这是一项独立工作，收益是 TTFT 公平性（两条都会
  在 ~128 s 附近出首 token）而不是吞吐（两条总计算量不变）。
- 更省事的方向：**减少每条请求的上下文**。按可用池子反推，4 并发各约 50K 是
  这台机器的现实上限。

### 旁证修正

整合稿 §3.3 曾把"8K 并发撞权重带宽墙"写成待裁决；同日 U0 已裁决为
**权重近似全额摊销**（C=2 步长 1.07×、C=4 步长 1.26×），本节的长上下文现象
与它不矛盾：200K 的串行是**准入/内存**问题，不是权重带宽问题。
