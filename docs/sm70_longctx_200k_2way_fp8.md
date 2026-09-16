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
| 2112 | 1024 | **段错误**（主机侧）。初版把它记成"Xid 31 + 段错误"是**误合并**：dmesg 里那条 Xid 属于九分钟前的另一跑（OOM kill 的 teardown 产物）。**更正见 §7** |

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
2. 就算放进去，分叉批量前向对空返回取下标 0 会崩（主机侧空指针，**不是 GPU bug**，已修，见 §7）。

### 处置

实验代码（env 开关）**已全部撤除**，工作区与 `src/` 保持干净，二进制已重建。
这两个问题都不该在没摸清 ragged batched prefill 生命周期的情况下继续试——
GPU MMU 故障说明是地址记账错了，不是参数没调好。

### 这台机器上的可选做法

- **200K × 1 并发：可行**（FP8 或 FP16 都能跑，FP16 单条实测 TTFT 127.6 s）。
- **200K × 2 并发：当前不可行**，卡在调度器准入，不在显存。
- 若坚持要 2 并发：**两个障碍都已清掉**——warmup 缓冲与准入预算的解耦，
  以及分叉批量前向的空返回（已修，提交 `56e3445f`）。用
  `FASTLLM_PREFILL_ROTATE=1` 实测该组合已真正并发（`inFlight=2` 97 轮、
  TTFT 253.3 / 254.6 s）。但**收益是 TTFT 公平性而不是吞吐**：Total time 与
  串行相同（约 255 s），而 #0 的 TTFT 从 127.4 s 变慢到 253.3 s。所以缺省
  关闭，只在并发长请求是常态时开启。详见
  `docs/sm70_prefill_rotation_review.md` §3.4。
- 更省事的方向：**减少每条请求的上下文**。按 0.90 的可用池子反推，4 并发各约 50K
  是当时算出的上限——**这条已被 §8 的 4×200K 实测推翻**。

### 旁证修正

整合稿 §3.3 曾把"8K 并发撞权重带宽墙"写成待裁决；同日 U0 已裁决为
**权重近似全额摊销**（C=2 步长 1.07×、C=4 步长 1.26×），本节的长上下文现象
与它不矛盾：200K 的串行是**准入/内存**问题，不是权重带宽问题。

## 7. 更正：Xid 31 与段错误不是同一件事（2026-09-15）

本文件初版把 14:xx 那跑的 dmesg 末 5 行读成"Xid 31（GPU MMU 故障）+ 段错误"，
并据此写下"ragged batched prefill 的地址记账有 bug"。**两处都错**：

1. **Xid 31 属于另一跑。** dmesg 里那条 Xid 的时间戳对应九分钟前的
   `--chunked_prefill_size 4096` 跑，它 `selected=2` 为 0、死在
   `cuda malloc failed ... dims=[4,1,2560,4]`（OOM）。崩溃那跑（PID 1884021，
   19:10:22）的**时间窗内没有 Xid**。
2. **合并的原因**：当时执行的是 `dmesg | tail -5`，而末 5 行恰好是九分钟前那条
   Xid 的 3 行加当下段错误的 2 行，于是两件事被读成一件。
3. **Xid 31 在本机的已知来源**是 OOM kill 的 teardown，不是数据故障：
   `sm70_npad_ar_chunk_landing_plan.md:554` 已经记录过同一现象
   （"进程被杀，teardown 产生 Xid 31 MMU fault"）。

**真实根因是主机侧空指针**：`runSplitBatchForward` 对空向量取下标 0。已修，
提交 `56e3445f`。**"需要修 GPU 地址记账"这个结论会让人去查一个不存在的 GPU
bug**，特此撤回。

## 8. 200K × 4 并发 × FP8 KV 实测（2026-09-16）

问题同开头那句："4×200K 现在能不能跑"。**结论：能跑完，但仍然是串行**——
四条请求全部被受理、全部出结果、exit 0，代价是总时长约等于 4 倍单条。

配置（4×V100-16GB / TP4 / 空载机器）：

```
--tokens 800000 --max_batch 4 --gpu_mem_ratio 0.98 --kv_cache_dtype fp8_e4m3
--low_gpu_mem --input_tokens 200000 --output_tokens 8 --batch 4
```

| 项 | 值 |
|---|---|
| 池 | 6250 页（`--tokens 800000` 全额满足），`AddPrefill` 上限 5000 页 |
| `availForKV` | **5.83 GB**（`reserved` 0.34 GB，`localKVPerPage` 1.05 MB）|
| 校准后 4 卡合计 gpuFree | 1304 MB（326 MB/卡）|
| TTFT | min 138.65 s、max **561.92 s**、avg 350.30 s |
| 总时长 | **562.18 s**（wall 598 s）|
| `prefillBlocked` | **0** |
| `selected=2 / 3 / 4` | 0 / 0 / 0 |
| 退出 | **exit 0**，无 OOM、无 illegal address |

三点读数对应三件事：

1. **内存这关过了**：6250 页 × 1.05 MB = 6.56 GB 的池装下了，页守卫一次没触发。
   `availForKV` 从 0.90 的 2.05 GB 抬到 5.83 GB，靠两笔叠加——`--gpu_mem_ratio
   0.98` 还回 1.36 GB/卡，`--low_gpu_mem` 再还回那份 2425 MB/卡 的 embedding
   全量副本。
2. **`--low_gpu_mem` 在这里是必需的**：同配置只去掉这个开关的对照跑 30 s 就
   `exit=1`，暖机期 `cudaErrorMemoryAllocation`（`--tokens 800000` 按 6250 页
   申请池子，直接把卡吃满）。
3. **调度这关没过**：`selected` 一次都没有 ≥2，TTFT max 561.92 s ≈ 4 × 138.65 s，
   四条 prefill 一条接一条走。所以 4 并发的收益是"都能跑"，不是"同时跑"。

本文档前面"按可用池子反推，4 并发各约 50K"那句作废：那是按 0.90 的 2.05 GB
池子反推的，现在的池子是它的 2.8 倍。要真正重叠仍然只有
`FASTLLM_PREFILL_ROTATE=1` 那条路，而它的收益是 TTFT 公平性，不是吞吐。
