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

### 3.7 多并发扩展性与硬件账（2026-09-15）

#### 一个 forward 最多装 2 条 chunk，第 3 条就崩

轮转的准入预算从 `2 x chunk` 放宽到 `4 x chunk` 后，`inFlight` 达到 **4**，
但立刻 **exit 134（abort）**：

```
inFlight 分布:  1 inFlight=4      exit=134（核心已转储）
```

配合前面的两次段错误（`inFlight=2` 时是另一处空返回 bug，已修），结论是：

| 一个 forward 里的 prefill chunk 数 | 结果 |
|---:|---|
| 1（缺省） | 正常，但完全串行 |
| **2** | **正常**（批量 decode/2 条 prefill 天天在走） |
| 3–4 | **abort** |

所以预算必须钉在 `2 x chunk`，这是实测出来的上限，不是保守取值。

#### 并发上限与池子的关系

| 场景 | 池子 | 实际并发 | 证据 |
|---|---|---:|---|
| 200K × 4（轮转开） | 3200 页 | **2** | 见 §3.4 第 3 条：R2 多等 116 s |
| 80K × 4（轮转开） | 3200 页 | **2** | TTFT 81715/164327 ms，`inFlight=2` |
| 80K × 4（轮转关） | 3200 页 | **1** | TTFT 41013/166200 ms，`inFlight=1` |

**80K × 4 的记忆算术上装得下**（4 × 640 = 2560 页 = 3200 的 80% 上限），但实测
只有 2 条并排。原因还没定位到具体代码位——可能是一处只处理 2 条的路径，也可能
是别处限制了同时 in-flight 的 prefill 数。**这一条是开放项，不要当成已解释。**

#### 5 张显卡能不能跑 4 × 200K：不能，但真实约束不是我最初算的那条

**先更正本文件 §3.7 初版的一处算错。** 初版用"总显存 − 权重 − 几个小预留"算出
每卡可给池子 + 激活 10.61 / 11.64 GB（TP4 / TP5），据此说 TP5 只补 1 GB。那个
算式**漏了约 7.37 GB/卡 的运行时常驻**，所以偏乐观了。

实测反推（TP4、FP8、`gpu_mem_ratio 0.97`）：

| 项 | 每卡 |
|---|---:|
| 卡总显存 | 16.93 GB |
| 建池前 free | **4.42 GB** |
| ├ 权重分片（20.56 / 4） | 5.14 GB |
| └ **其余常驻** | **7.37 GB** |
| 池子实得 | **4.10 GB**（3907 页 × 1.05 MB）|
| 留给激活 | **0.32 GB** |

**池子上限实测**：`--tokens 500000` → 3907 页（exit 0）；`700000` / `900000` /
`1100000` → 全部 `exit 1`（`cuda malloc failed`）。所以每卡池子硬上限约
**3907 页 = 4.10 GB**，与 `availForKV` 报的 3.24 GB 基本一致。

需求侧：4 × 200K 需要 6252 页 = **6.56 GB** KV（FP8），除以 80% 上限得池子
**7815 页 = 8.21 GB**。

| 布局 | 每卡池子上限 | 4×200K 需要 | 缺口 |
|---|---:|---:|---:|
| TP4 | 3907 页 | 7815 页 | **3908 页** |
| TP5（第 5 卡省 1.03 GB 权重） | 约 4886 页 | 7815 页 | **约 2929 页** |

**第 5 张卡在权重上省 1.03 GB/卡，够不到缺口的量级。**

再补两点：

- **5 卡的聚合显存确实够**（80 GB 对 20.56 权重 + 26.24 KV + 常驻，约 59 GB），
  但约束是**每卡**的池子上限，不是总量。总量够而每卡不够，加卡不会自动解决。
- **只加卡不改布局，每卡 KV 也不会变小**：GQA 是 4 个 kv head，TP4 已经切成
  **每卡 1 个 head**，5 卡整除不了 4 个 head，所以第 5 卡不减少每卡 KV 页数。

**真正能改这条约束的方向**（按代价排）：

1. **修"一个 forward 最多 2 条 chunk"那条路径**（§3.7 第一小节）。它不增加内存，
   但能让已装得下的组合（如 80K × 4）真正四路并行。**不解决 4 × 200K 的内存。**
2. **砍掉那 7.37 GB/卡 的常驻**。它比权重还大，是当前最大的未知项；本轮只反推出
   了它的总量，没定位到具体分配点。要解决 4 × 200K，这一项比加卡更有希望。
3. **context parallel**：CP2 让每卡只存 100K，4 × 200K 的 KV 减半到 3.28 GB，
   `TP4 + CP2` 需要 8 张卡。但本机 NVLink 全 inactive、走 PCIe，200K 下每层
   每卡要换 328 MB、16 层约 5.2 GB/step，几十到几百毫秒每步，而整个 decode 步
   才 12–17 ms。**加卡也会先撞 PCIe 墙。**

### 3.8 每卡常驻的归因（2026-09-15 实测，含一处自我更正）

给启动路径加了内存检查点（`FASTLLM_MEM_TRACE=1`）与逐缓冲转储
（`FASTLLM_MEM_POOL_DUMP=1`）后，账目如下。

**权重：5.37 GB/卡。** `AutoWarmup` 入口 4 卡合计 gpuFree=43100 MB，
`64576 − 43100 = 21476 MB`，与 20.56 GB ÷ 4 吻合（NVFP4 每参数 0.5625 字节）。

**FastLLM 的大缓冲池：10.7 GB/卡，但其中可回收的只有 0.36 GB。** 逐缓冲拆分：

| 项 | dev 0 | dev 3 |
|---|---:|---:|
| `bigPool` 总计 | 11075 MB | 10962 MB |
| `bigBusy`（在用） | **10711 MB** | 10708 MB |
| `bigGraphPinned` | 0 MB | 0 MB |
| **`bigIdleUnpinned`（可回收）** | **363 MB** | **254 MB** |

尺寸直方图（dev 0，按缓冲大小分档）：

| 档位 | 缓冲数 | busy | idle |
|---|---:|---:|---:|
| ≤2 MB | 2 | 0 MB | 3 MB |
| ≤8 MB | 149 | 641 MB | 112 MB |
| **≤32 MB** | **396** | **6991 MB** | 215 MB |
| ≤128 MB | 2 | 48 MB | 34 MB |
| **>512 MB** | **2** | **3031 MB** | 0 MB |

**更正本文件初版的一处措辞。** 初版说"其余 6.3 GB 是预热期留下的池化缓冲"，
并暗示它们是空闲缓存。**实测不成立**：池里几乎全是 `busy`，`idleUnpinned` 只有
254–363 MB。而且初版那 7.37 GB 是拿"总显存 − 权重"倒推的，里面本来就含 KV 页池，
所以"7.37 GB 去向不明"这个说法本身也不准确。

**正确的说法是**：每卡 16.93 GB ≈ 权重 5.37 GB + 大缓冲池 10.7 GB（其中 KV 页
3.36 GB）+ 少量空闲/碎片。池里 10.7 GB 的 busy 由两部分构成——几千个页级缓冲
（≤32 MB 档，396 个缓冲共 6991 MB，与该档平均 17.6 MB 吻合，是各层各自的页）
和 2 个 >512 MB 的大缓冲（3031 MB，可能是打包页池或某个大工作区）。

**回收路径是否存在**：存在（`FastllmCudaRetryMallocAfterReleasingIdle` →
`ReleaseIdleCachedBuffersForDevice`，只在分配失败后触发），但**它只能放出
0.36 GB 这个量级**，因为 `busy` 与 `graphPins > 0` 的缓冲它都不动。这解释了
为什么 `--tokens` 从 500000 加到 700000 就直接 `cuda malloc failed`：不是回收
没生效，而是**没有可回收的东西**。

**对 4 × 200K 的含义（本节的结论）**：那 4 GB 的缺口**不在这 10.7 GB 里**——
它是 busy 的，不是闲着的。所以"砍掉常驻"这条路**在本轮证据下是死的**，我之前
说它"最该查"的判断要收回。要腾出 4 GB，只有减少请求上下文、或 context parallel
（每卡只存一部分序列，但会撞 PCIe），或者换卡。

### 3.9 代码处置补充### 3.9 代码处置补充

- **新增 `FASTLLM_MEM_TRACE=1`**（缺省关）：让启动期打印每阶段
  `cudaMemGetInfo` 与池统计。它存在的理由很具体——`verbose` 是在模型构造**之后**
  才设置的，所以通过它看不到任何标定过程；这个开关补上了那段可观测性。
- 准入预算**保持 `2 x chunk`**（实测上限：3 条以上 abort）。
- 预热缓冲与准入预算**已解耦**（`servingPrefillTokenLimit = GetChunkedPrefillSize()`）：
  放宽预算时预热不再同比放大，否则 4x 预算会在第一条请求之前就 OOM
  （实测 `cuda malloc failed dims=[1,8192,8704]`、`gpuFree 11 MB`）。
  缺省下这个值本来就是 chunk 尺寸，所以**缺省行为不变**（回归见下）。

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

## 7. "10.7 GB 池子能不能瘦" 的逐项实测（2026-09-15）

背景：每卡 16.93 GB 里有约 10.7 GB 落在 FastLLM 自己的大缓冲池（`bigPool`），
而 KV 页池只占其中 3.36 GB。本轮逐项排查"能不能把这 10.7 GB 瘦下来"，
**六条假设里五条被实测挡回，第六条（关 CUDA embedding）当日被误判，09-16 复测翻案**。

| 假设 | 探针 | 结果 |
|---|---|---|
| 是 KV 页池 | `--tokens 2048`（16 页）vs `409600`（3200 页）| **`bigBusy` 两者都是 10711 MB**，与 KV 池无关 |
| 是 CUDA Graph 预留 | `FASTLLM_CUDA_GRAPH=0` | **仍 10711 MB**，与图无关 |
| 随 batch 放大 | `--max_batch 1/2/4` | 10708 / 10711 / 10714 MB，几乎不变 |
| 是 embedding 运行时副本 | `--cuda_embedding`（该写法未生效）| 不变 —— **这次探针无效** |
| 空闲缓冲可回收给更大池子 | `FASTLLM_MEM_POOL_DUMP=1` | `bigIdleUnpinned` 仅 **254–363 MB** |
| 关 CUDA embedding 省 2425 MB | `--low_gpu_mem` | **成立**，每卡还回 2425 MB（09-16 复测推翻初判，见下）|

### 两个大缓冲的身份

一条错误信息直接指认了它：

```
FastLLM fatal CUDA allocation error: cuda malloc failed in Data::ToDevice CPU->CUDA.
  requestBytes = 2542796800, dataType = float16,
  dims = [248320, 5120], name = model.language_model.embed_tokens...
```

`2542796800 B = 2425.0 MB`，正是 `[vocab 248320, hidden 5120]` FP16 **未分片**的
embedding。另一个 606.2 MB 恰好是它的 1/4，即 **TP4 分片**。也就是说每张卡上
同时有**全量副本与 TP 分片两份**。分配点：

```cpp
// qwen3_5.cpp:16125
if (tensorParallel && !useCpuEmbedding) {
    PrepareMultiCudaReplicatedData(weight[language_prefix + "embed_tokens.weight"],
                                   devices, true);   // 每卡一份全量
}
```

`useCpuEmbedding = !GetCudaEmbeddingRequested() || GetLowMemMode()`（15276）。

### "关掉它省 2425 MB"：09-16 复测成立，初判被推翻

09-15 首测以为这条路更糟：`--low_gpu_mem` 那跑在"权重加载后"只报 4 卡合计
4032 MB，随后一串 `cudaErrorIllegalAddress`。09-16 在空载机器上按同一配置
（`--tokens 409600 --max_batch 2 --batch 2 --input_tokens 8192`，开
`FASTLLM_MEM_TRACE=1`）两臂各跑一次：

| 臂 | 权重加载后 gpuFree | 校准后 gpuFree | 结果 |
|---|---:|---:|---|
| `--cuda_embedding`（默认） | 43100 MB | 未到 | **exit=1**，暖机期 23 MB sidecar 分配 `cudaErrorMemoryAllocation` |
| `--low_gpu_mem` | **43100 MB** | 1014 MB | **exit=0**，完整出报告（sha `076b2552…`） |

两臂在"权重加载后"这一点**读数完全相同**（都是 43100 MB），说明该打印点落在
embedding 副本落地**之前**，本来就看不见那 2425 MB；换句话说这个探针测不出
`--low_gpu_mem` 的影响。旧测的 4032 MB 也因此不是这个开关造成的：那一跑与另一份
4 卡作业（`emb_off`，08:25 起、900 s 超时后被杀）时间重叠，被占掉的约
（43100 − 4032）/ 4 ≈ 9.8 GB/卡落进了读数；随后的崩溃是内核态 `Xid 13`
（warp 越界地址，dmesg 08:28:37）冒出来的 `cudaErrorIllegalAddress`，不是 OOM
（该日志里 error 700 出现 6 次，error 2 一次都没有）。

结论更正：关掉 CUDA embedding **确实**把每卡 2425 MB 还回来，这条路是通的。

09-16 的完整复测（每臂各两次，空载机器，8K 输入 / 64 输出 / C=1 / TP4 /
`--tokens 16384 --max_batch 1`）：

| 臂 | 每卡空闲 | availForKV | TTFT | decode | sha256 | 退出 |
|---|---:|---:|---:|---:|---|---|
| `--cuda_embedding` ×2 | 4.45 GB | 2.09 GB | 3155.69 / 3147.40 ms | 86.74 / 87.02 tok/s | `b960451a…` | exit 0，44–45 s |
| `--low_gpu_mem` ×2 | 7.00 GB | 4.63 GB | 3188.43 / 3185.37 ms | 89.67 / 89.64 tok/s | `b960451a…` | exit 0，40–42 s |

最干净的一行是"校准后 4 卡合计 gpuFree"：默认臂两次都是 16034 MB，
`--low_gpu_mem` 臂两次都是 25738 MB，差 9704 MB ÷ 4 = **2426 MB/卡**，正好是那份
未分片副本（2425 MB）。两次低显存跑的 decode 都比基线高 2.6–3.4%，TTFT 高
1.0–1.2%，六次报告的 sha256 全同。

同一个 3200 页的配置（`--tokens 409600 --max_batch 2`）在**当前构建**下方向是反的：
默认臂两次都死在暖机期（exit=1，23 MB sidecar 分配 `cudaErrorMemoryAllocation`，
gpuFree 只剩 19/7/21 MB），`--low_gpu_mem` 臂两次都跑完（exit=0，校准后 1014 MB，
sha `076b2552…`）。07:12 那次默认臂是跑通的，但 `libfastllm_tools.so` 在 08:25
重装过，两次不是同一份二进制，所以这条只作为待查项记下，不作为结论。

### 结论

1. 那 10.7 GB **不是浪费**：它与 KV 数量、CUDA Graph、batch 都无关，是固定的
   **运行时工作集**，其中两个最大块是 embedding 的全量副本与 TP 分片。
2. **空闲缓冲可回收的只有 254–363 MB**，但 embedding 的**全量副本**
   （2425 MB/卡）可以靠 `--low_gpu_mem` 真正拿回来——09-15 的初判被 09-16 复测
   推翻，见上。
3. 因此 4 × 200K 只能靠**减少每条请求的上下文**，或**换硬件/布局**
   （context parallel + NVLink）。省下这 2425 MB/卡换来的是更大的 KV 页池
   （实测 availForKV 2.09 → 4.63 GB），不改变这个结论。

### 本轮新增的观测能力

- `FASTLLM_MEM_TRACE=1`：启动期每阶段打印 `cudaMemGetInfo` 与池统计。
  存在的理由很具体——`verbose` 在模型构造**之后**才设置，通过它看不到任何标定过程。
- `FASTLLM_MEM_POOL_DUMP=1`：逐缓冲拆分 `busy / graphPinned / idleUnpinned`，
  外加尺寸直方图与"按尺寸分组"（同尺寸重复 = 同一个分配点）。
  这次正是它推翻了"池里是空闲缓存"这个说法。
