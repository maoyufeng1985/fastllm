# 长上下文并发：prefill 串行入队优化方案

日期：2026-09-14（生产路径实测 2026-09-15）
范围：Qwen3.8-27B-NVFP4 / SM70 / TP4 / no-MTP。生产 auto
`FASTLLM_GPU_TOKEN_HANDOFF=1` 走 `Qwen35MTPLoop`，不是 `RunNewMainLoop`。
`UseModelSpecificScheduler()` 在 MTP、DFlash、或 handoff 时为真。
结论先行：80K C=2 的真实瓶颈是长 prompt 在调度器里被整段放行。一条 81920 token
的 forward 独占 GPU 41.6 s，这期间另一条请求几乎停止 decode。不是 kernel 慢，
也不是带宽瓶颈。选批分块之后，#0 在 #1 prefill 期间产出 40 个 decode token
（每 2048 让位一次）。两条 80K 的 prefill 仍然串行，因为 qwen3_5 的
`GetBatchedPrefillTokenLimit()` 是 2048。

## 1. 现象

`docs/sm70_concurrency_port_plan.md` §16.2 的实测（graph on，4×V100-16GB）：

| C=2 80K | 值 |
| --- | ---: |
| TTFT #0 / #1 | 41.50 s / 83.08 s |
| common window | 83.09 s + 4.70 s |
| 窗口内合计 | 108.46 tok/s（两条各 54.23） |
| 先拿到首 token 的那条请求，在随后 41.6 s 内的 decode 产出 | **至多 1 token** |

最后一行不是直接观察到的，是从工具已有的数字推出来的，而且两个口径互相闭合。
`request #0 in window` 和 `request #1 in window` **都是 54.23 tok/s**，窗口 4.70 s，
乘出来各 254.9 个 token，而每条请求的 decode 总量都只有 255 个（输出 256 个，
减掉作为 prefill 结果的第一个）。254.9 贴到 255，说明**两条请求的 decode token
几乎全在窗口内**，窗口之前基本没有产出。另一个独立口径吻合：
`TPOP max = 181.51 ms/token` 正好是先到那条的全跨度平均 `(87.79 − 41.50) / 255`。

注意是「至多 1」不是「恰好 0」。让位机制会在长 prefill 之后强制插入一轮 decode，
所以先到的请求很可能拿到 1 个 token，随后被卡住。两种读数都不改变结论。

于是这个 workload 的形态是「41.6 s 纯 prefill + 4.7 s 全速 decode」。
decode 本身不慢，108.46 的聚合相对 C=1 的 61.16 是 1.77x。

## 2. 根因

### 2.1 长 prompt 被整段放行，没有分块

`src/models/basellm.cpp` 的 `RunNewMainLoop`（1296 行起）里，唯一的
`prefillChunkSize` 使用点是 1309 行，随后传进 1829 行的
`currentPrefillTokenLimit`，再在 2244 行的选批逻辑里做门控：

```2244:2249:src/models/basellm.cpp
                        if (thisLen > prefillChunkSize) {
                            if (seqLens.size() > 0) {
                                continue;
                            }
                        } else {
                            if (prefillTokenCount + thisLen > currentPrefillTokenLimit && seqLens.size() > 0) {
                                continue;
                            }
                        }
                        prefillTokenCount += thisLen;
```

`thisLen` 是 `ctx->currentTokens.size()`（2210 行），而请求创建时
`currentTokens = inputTokens`（3184 行），是**整条 prompt**。

于是对 80K prompt：

- `thisLen = 81920`，远大于 `prefillChunkSize`（qwen3_5 的
  `defaultChunkedPrefillSize = 2048`，见 `src/models/qwen3_5.cpp:8838`）。
- 走 `thisLen > prefillChunkSize` 分支。当 `seqLens` 还空着（这是本轮第一个
  被选中的请求）**不 continue**，整条 81920 被放行。
- `prefillTokenCount += 81920`。

注意这个分支的语义是「超过 chunk 的请求，只有在它是唯一候选时才允许，
并且是**整条**跑」。它没有把请求切成 chunk，只是允许一次超长 forward。

### 2.2 为什么超长 prompt 会串行

选批顺序在 1727 行按 prompt 长度**降序**排：

```1727:1727:src/models/basellm.cpp
                orders.push_back({-(int)it.second->currentTokens.size(), it.first, it.second});
```

C=2 两条都是 80K，于是：

1. 第一条被选中，`seqLens.size()` 变为 1，`prefillTokenCount = 81920`。
2. 第二条再次进入循环时 `seqLens.size() > 0` 且 `thisLen > prefillChunkSize`，
   命中 `continue`，被跳过。
3. 这一轮 forward 只有第一条，跑完整 81920。

下一条要等这条结束后才可能被选中。两条 80K 因此**必然串行**，
TTFT 间隔正好一个 41.6 s 的 full-length prefill。这也解释了
TTFT #1 − TTFT #0 = 41.58 s ≈ 一次 80K forward 的耗时。

### 2.3 为什么现成的 interleave 机制救不了

代码里**已经有**一套 add-prefill 让位机制（1310–1337 行设置
`interleaveActivePrefill`，1823–1831 行据此在每轮 add-prefill 后让一轮 decode）。
它对 qwen3_5 + `maxBatch > 1` 默认开启，所以 8K 场景是生效的。

但它让位的**粒度是一轮 forward**。对 80K：

- prefill 只占 **1** 轮 forward（因为整段放行），
- 这一轮本身长 41.6 s。

也就是说，没有「两轮之间」可供插入 decode。让位机制没坏，是长 forward
把它架空了。这与 §16.2 的观察一致：8K 的 `C=4` 聚合被压到 47 tok/s（有多轮，
让位生效但收益有限），而 80K 的 decode 被压到几乎停滞。

### 2.4 旁证：老引擎里其实写了分块循环

`LaunchResponseTokens`（2699 行起，**旧引擎**路径）里有一段真正的分块循环：

```3046:3066:src/models/basellm.cpp
                                if (seqLens[0] > first) {
                                    int len = seqLens[0];
                                    for (int st = 0; st < len; ) {
                                        ...
                                        int curLen = std::min(st == 0 ? first : part, len - st);
                                        Split(inputIds, 1, st, st + curLen, curInput);
                                        Split(*positionIds[0], 1, st, st + curLen, curPositionIds);
                                        ret = std::vector <int> {model->Forward(curInput, Data(), curPositionIds,
                                            *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                        st += curLen;
                                    }
                                }
```

但 qwen3_5 的 `use_new_engine = true`（`qwen3_5.cpp:8837`），
走的是 `RunNewMainLoop`，**根本不会执行这段**。所以分块能力在旧引擎存在、
在新引擎缺失。这是「新引擎还没补齐旧引擎已有的长 prefill 分块」，
不是需要新发明的算法。

## 3. 目标

1. 80K C=2：先到的那条请求在 #1 prefill 期间不再停止产出，常见 decode 窗口内的
   聚合吞吐不低于当前 108.46 tok/s（不牺牲稳态），且 #0 的首阶段有持续推进。
   （实施后实测：让位门达成。108.46 当时经同树 chunk off 零让位对照证明是
   QPN2-on 时代 kernel 速率，在 QPN2-off 的树上不可达且与调度器无关，见
   §5.1/§9。2026-09-15 更正：QPN2 已可缺省开（warmup 崩溃已修），
   common window 实测 115.39 tok/s 且保留 40 token 让位，本门在 QPN2-on
   树上达标。）
2. 长 prompt 不再垄断 GPU：单轮 forward 的 token 数受 `prefillChunkSize` 约束。
3. C=1 与 8K 场景的数字不回退。
4. 不改变权重格式、不改 checkpoint、不动 CUDA Graph 契约。

非目标：不做 prefill/decode 同批混合（vLLM 式 mixed batch），那是更大的改动，
本方案先拿到「分块 + 让位」这一层收益。

## 4. 方案（PR-A 落地）

结论先说：**不要去改 2500 行那套「单请求内部分块循环」。** 它已经能把 80K 切成
2048，但整段锁在 `forwardLocker` 里，40 个 chunk 连着跑完才还锁。让位机制插不进去。
PR-A 要做的是**选批分块**：每轮只提交一个 chunk，forward 结束就还锁，让位自然生效。

对照 1Cat：生产 `--enable-chunked-prefill --max-num-batched-tokens 2048`。本方案
只要「分块 + 让位」，**不做** vLLM 式 prefill/decode 同批混合。

### 4.0 现有代码里已经有、但不能用的两层

| 层 | 位置 | 做什么 | 为什么救不了 80K C=2 |
|---|---|---|---|
| 选批门控 | `basellm.cpp:2240` | `thisLen > prefillChunkSize` 时，只有本轮第一个请求能进，且**整段**进 | 没有切，只是互斥 |
| 单请求内循环 | `basellm.cpp:2500` `seqLens.size()==1 && seqLens[0] > prefillChunkSize` | 把已经组好的 80K `inputIds` 再 `Split` 成 2048 | 锁着 `forwardLocker`，不让位；C=2 的第二条仍被 2240 挡在门外 |
| 旧引擎循环 | `LaunchResponseTokens` ~3046 | 真正的分块 Forward | qwen3_5 `use_new_engine=true`，走不到 |

qwen3_5 默认 `defaultChunkedPrefillSize = 2048`。`tools/fastllm_pytools/util.py`
的 `_configure_qwen35_auto_fast_paths` 在 unset 时把
`FASTLLM_GPU_TOKEN_HANDOFF=1`，于是 no-MTP 生产路径是 `Qwen35MTPLoop`。
`FASTLLM_GPU_TOKEN_HANDOFF=0` 才进 `RunNewMainLoop`。两份循环都接了同一套
`include/models/longPrefillChunk.h` remaining-count。真 MTP / DFlash
（`mtpDraftsPerStep > 0` 或 DFlash）仍整段放行，好让 22964 的内循环给
draft KV 做 seed。`RunNewMainLoop` 没有 GPU token handoff，所以不能把
no-MTP handoff 改道到那条循环。

### 4.1 主改动：选批时只取一个 chunk，forward 后再前进状态

改 `2210` 起的选批，不改 2500 的内循环。内循环在选批分块生效后自然失效
（`seqLens[0]` 不再 > 2048），可以留着当 `FASTLLM_LONG_PREFILL_CHUNK=0` 的回退。

#### 状态机（这是本方案真正要发明的一点）

现有约定：

- `preTokens == 0` → 还在 prefill，走 `isPrompt=1`
- `preTokens > 0` → 已经 decode，走 `isPrompt=0`
- forward 结束后无条件 `currentTokens.assign(1, curRet)`，把剩余 prompt 清掉

若中间 chunk 把 `preTokens` 加成 2048，下一轮会被 `isPrompt && preTokens != 0`
跳过，当成 decode；再被 `assign(1, curRet)` 把剩下的 79K prompt 清掉。**这是
PR-A 最大的正确性坑，plan 初版没写死。**

新增一个显式标志，不要复用 `preTokens`：

```
ctx->intParams["prefill_remaining"] = 还没喂进模型的 prompt token 数
```

- 请求创建时不设；第一次选中长 prompt 时设为 `currentTokens.size()`（prefix
  restore 之后的剩余量）。
- 选批：`isPrompt` 当且仅当 `preTokens == 0 || prefill_remaining > 0`。
  即：正在分块的请求继续走 prefill 分支，即使 `preTokens` 已经被加过。
- 每个 chunk 的 `curLen = min(prefill_remaining, prefillChunkSize, currentPrefillTokenLimit - prefillTokenCount)`。
- **只把 `currentTokens[0, curLen)` 交给 `FillLLMInputs`**，不要把整条剩余 prompt
  塞进本轮。
- forward 结束后：
  - 若 `prefill_remaining - curLen > 0`：erase 头部 `curLen`；
    `prefill_remaining -= curLen`；`cacheLen += curLen`（若模型用 cacheLen 记
    已写入 KV 的长度）；**丢掉本轮 `ret[i]`，不 push 到 `resultTokenQueue`，
    不 `allTokens.push_back`，不 `curTokens++`，不判 eos**。
    中间 chunk 的 lm_head 输出不是生成 token（qwen3_5 在
    `isIntermediateChunkedPrefill` 下已经跳过 head，`ret` 无意义）。
  - 若这是最后一个 chunk：erase 头部 `curLen`，清 `prefill_remaining`，
    **这时才** `assign(1, curRet)` / push / 进入 decode。
- `preTokens` 仍然累加 `curLen`，给 verbose / 统计用，但不再当 isPrompt 判据。

更干净的替代：单独加 `ResponseContext::prefillRemaining`。`intParams` 少改头文件，
字段更不容易漏。二选一，PR 里只留一种。

#### `promptLen` / position（不能按剩余量设）

`basellm::FillLLMInputs`：

```
vpids[i] = promptLen - seqLen + i
```

`promptLen` 必须是 **cacheLen + 本 chunk 之前已消费 + 本 chunk 长度**，也就是
「到本 chunk 末尾为止的绝对位置」，不是剩余 prompt 长度。

- 第一个 chunk：`promptLen = cacheLen + curLen`，`index = 0`
- 后续 chunk：`promptLen = cacheLen + alreadyConsumed + curLen`，`index++`
- qwen3_5 `NeedAttentionMask` 恒为 false，attentionMask 为空，因果由 kernel 做。
  别的模型若 `NeedAttentionMask=true`，mask 宽是 `promptLen` 不是 `seqLen`，
  必须按绝对位置填，否则第二块会把历史当成不可见。

现有 2290 行 `promptLen = cacheLen + currentTokens.size()` 在切短 `currentTokens`
之后会变成「cacheLen + 剩余全长」，第二个 chunk 起位置会跳。必须改成上面的公式。

#### page 记账按 `curLen`

`thisPages`、`collectPrefillPageNeeds(ctx, thisLen)`、`pagesLimit` 全部用
`curLen`，不要用剩余全长。`addManagerPageNeed` 对 sliding-window 层已经按
`retained + prefillChunkSize` 封顶，传 `curLen` 与它一致。

80K 仍按整段峰值占页（KV 最终还是 80K），分块不降低峰值，只降低单轮
`AddPrefill` 的增量。`pagesLimit = totalPages * 4/5` 的软门继续按增量检查。

#### 选批顺序与让位

降序（长 prompt 优先）保持。分块后每条每轮只占 2048，`currentPrefillTokenLimit`
（idle 突发 = `batchedPrefillTokenLimit`，add-prefill = `min(..., 8192)`）
可以在同一轮塞进第二条长 prompt 的一个 chunk——这是「交替推进」，
**仍然不是** mixed batch（`isPrompt==0 && seqLens.size()>0` 会跳过 decode，
1837 行不动）。

`interleaveActivePrefill` 不动。80K 从 1 轮变成 40 轮，
`forceDecodeThisIteration` 每轮 add-prefill 后插一轮 decode，正是要的。

idle 突发窗口（1756 行，等齐同一波 HTTP）也不动。qwen3_5 的
`GetBatchedPrefillTokenLimit()` 是 2048，所以两条 2048 chunk 不能同批。
分块插入 decode，不重叠两条 80K prefill。

### 4.2 明确不改

- 2500 行内循环：回退路径保留。选批分块生效后进不去。
- CUDA Graph：仍只收 `dims[1]==1`。chunk 是 eager prefill。
- `FASTLLM_PAGED_CUBLAS_CHUNK`：算子分块，与调度正交。80K 继续 2048。
- mixed batch / `max_num_partial_prefills`：PR-C，本方案非目标。
- 真 MTP / DFlash 的内循环 seed：`mtpDraftsPerStep > 0` 或 DFlash 时
  仍整段放行。选批分块只开在 no-MTP handoff。

### 4.3 开关

```
FASTLLM_LONG_PREFILL_CHUNK
  缺省 / unset / 非 0 ：启用选批分块
  0                    ：回到现在的整段放行 + 2500 内循环
```

`prefillChunkSize <= 0` 同样禁用。chunk 尺寸仍走 `GetChunkedPrefillSize()` /
`SetChunkedPrefillSize`，不新增第二个尺寸旋钮。

### 4.4 伪代码（选批内核）

```cpp
const bool longPrefillChunk =
    prefillChunkSize > 0 && env_is_not_zero("FASTLLM_LONG_PREFILL_CHUNK");

// isPrompt 判定，替换 1862/1865：
const bool stillPrefilling =
    ctx->preTokens == 0 ||
    (longPrefillChunk && ctx->intParams["prefill_remaining"] > 0);
if (isPrompt && !stillPrefilling) continue;
if (!isPrompt && stillPrefilling) continue;

// 2210 起：
int remaining = (int)ctx->currentTokens.size();
if (longPrefillChunk && ctx->intParams.count("prefill_remaining")) {
    remaining = ctx->intParams["prefill_remaining"];
}
int curLen = remaining;
if (longPrefillChunk) {
    curLen = std::min(curLen, prefillChunkSize);
    if (seqLens.size() > 0) {
        curLen = std::min(curLen,
            currentPrefillTokenLimit - prefillTokenCount);
        if (curLen <= 0) continue;
    }
}
// thisPages / collectPrefillPageNeeds / pagesLimit 用 curLen
// FillLLMInputs 只用 currentTokens[0, curLen)
// promptLen = cacheLen + (原剩余 - remaining) + curLen
prefillTokenCount += curLen;
```

forward 后（2633 循环里，对 `seqLens[i] > 1` 且 `prefill_remaining > curLen` 的项
走「中间 chunk」分支，其余走现有 decode 提交）。

中间 chunk 设 `isIntermediateChunkedPrefill`（现有 flag，qwen3_5 用来跳 lm_head）。
选批分块后一个 batch 可能有**两条**中间 chunk，现有 flag 是进程级 bool，只在
2500 单请求内循环里设。PR-A 要么：

1. greedy 且 batch 全是中间 chunk 时才设（C=2 两条都没跑完时成立）；
2. 或者暂时不设，多算一次 lm_head，正确性优先。

推荐 1，C=2 80K 的稳态形态正好是「两轮都是中间 chunk 交替 / 一轮 decode」。

### 4.5 正确性门（先于端到端）

单请求 greedy，同一 prompt 跑 `FASTLLM_LONG_PREFILL_CHUNK=0/1`，token sha256
必须相同。覆盖：

- 短 prompt（< 2048）：两路都不切，应当 bit-identical 且无额外 forward
- 恰好 2048、2049、4096、81920
- prefix cache 命中一段后再分块（`cacheLen > 0`）
- C=1 与 C=2

position 抽查：第二 chunk 的 `positionIds[0] == cacheLen + 2048`，不是 0。

### 4.2 让位机制沿用，不改语义

`interleaveActivePrefill` / `activePrefillNeedsDecode` 那套保持不动。
分块之后 prefill 从 1 轮变成 40 轮（80K / 2048），让位机制自然获得
「两轮之间」的插入点。这是本方案最省的一处：**不新增调度概念，
只把让位机制的目标对象从「一整条 prompt」变成「一个 chunk」**。

### 4.3 兜底与开关

见 §4.3。`prefillChunkSize <= 0` 时禁用分块，保持旧行为。

## 5. 验收

### 5.1 判据（用 `Batch decode (common window)`）

| 场景 | 指标 | 通过标准 | 实测（graph on，缺省 QPN2 on，`Qwen35MTPLoop`） |
| --- | --- | --- | --- |
| 80K C=2 | 请求 #0 在 #1 prefill 期间的 decode 产出 | **> 0 且持续**（改前为 0） | **40 token**（81920/2048，每 chunk 让位 1 次） |
| 80K C=2 | common window 聚合 | ≥ 108.46 tok/s（不回退） | 75.83 tok/s（QPN2 off，graph on）。108.46 在 QPN2-off 的树上不可达，与调度器无关：同树 chunk off 零让位对照实测 78.35（窗口 6.50 s，509 token，per-request 39.32/39.25，sha256 同 `ea2857f4`）。让位只解释 −2.5（78.35→75.83），其余 −30 是 batch-2 decode 速率本身（QPN2-off 39.3 vs 基线 54.23）。且按 `_common_decode_window` 定义（窗口 last TTFT→batch_end），聚合天生奖励零让位。2026-09-15 更正：QPN2 缺省开后实测 **115.39 tok/s 且保留 40 token 让位，达标**（当日 16:12 在当前含 combine 二进制上复测；combine 之前的初版值是 110.10） |
| 80K C=2 | TTFT #1 | ≤ 83.08 s（不回退） | 82.81 s ✓。83.08 本身就是守恒值 2×80K÷~1975 tok/s（对照：163840÷1996.83=82.05≈82.81；基线 163840÷1958.80=83.65≈83.09）。「明显小于」⟺ prefill 吞吐 ≫2000 tok/s，是 kernel 目标，调度器（含 PR-C）动不了；且等长 prompt 下与让位门互斥（yield>0 ⟹ #1 的 prefill 排在 #0 之后，`ShouldForceDecodeThisIteration` 要 decodeActive>0） |
| 80K C=1 | decode | ≥ 61.16 tok/s（不回退） | 56.48 tok/s（QPN2 off，graph on）。chunk off 对照 56.64，所以不是调度器回退。2026-09-15 更正：QPN2 缺省开后实测 **73.51 tok/s，达标**（61.16 是 QPN2-on 基线，原 QPN2-off 的树上不可测；73.51 同为当前含 combine 二进制复测值，初版 67.18） |
| 8K C=4 | 聚合 | ≥ 当前值（不回退） | graph-on 129.27 tok/s（QPN2 off）；before last TTFT 12/8/4/0；sha256 见下行 |
| 任意 | greedy token 流 sha256 | 与改动前一致 | 8K C=2 `5bfaac89` 对 `FASTLLM_LONG_PREFILL_CHUNK=0`（同二进制，C 单元保 sha 已证）；80K C=1 `9bb0aa71` 对 chunk off；80K C=2 `ea2857f4` 与 handoff-off 同。**口径警告（2026-09-15）**：`_token_stream_hash` 把全部生成 token 一起哈希，所以 sha 只在**同 `--output_tokens`** 下可比。曾把 `0e75bdf6`/`874972b7`（out=64）与 `5bfaac89`/`5b633030`（out=256）当成"跨二进制漂移"，那是口径错，已撤销；长度对齐后跨二进制稳定（80K C=1/C=2 各自横跨两次二进制一致）。详见落地方案 Appendix E |

### 5.2 必须同时盯的次生指标

- prefill 总吞吐不能掉：分块后每 chunk 的 GEMM 形状变小，
  `1971 tok/s` 的 prefill 速率可能下降。要同时报 TTFT #0/C=1 prefill，
  设一个可接受下限（例如不低于 `1971 * 0.85`）。
- 显存：分块不改变 KV 占用峰值，但要确认 `compactBlock` 与
  `prefillChunk` 相关的容量预留（basellm.cpp:4356-4364）在新路径下仍
  成立。已核：该块 `git diff` 未触及，`compactBlock = max(compactBlock,
  prefillChunk)` 在新路径下成立。

### 5.3 复现命令

```sh
FASTLLM_PAGED_CUBLAS_CHUNK=2048 \
PYTHONPATH=build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 4 --tokens 167936 \
  --dtype auto --enable_thinking false --prefix_cache false \
  --input_tokens 81920 --output_tokens 256 --batch 2 \
  --warmup 0 --temperature 0 --top_k 1
```

不要设 `FASTLLM_GPU_TOKEN_HANDOFF=0`。unset 时 auto-handoff 会把它打开，
调度器必须打出 `scheduler=Qwen35MTPLoop`。本树 QPN2 sidecar 会在 warmup
崩 cublas，所以复现关掉它。graph 由 auto fast path 打开。

判据看 `request #N before last TTFT`（让位）和 `Batch decode (common window)`
（速率）。**不要**再用 `Batch decode after TTFT` 判断入队改善，它分不清
prefill 与 decode。

零让位对照（§5.1 拆账用）：同命令加 `FASTLLM_LONG_PREFILL_CHUNK=0`。
预期：让位 0/1，common window ≈78（本树 batch-2 decode 速率），
sha256 仍 `ea2857f4`。

## 6. 风险

| 风险 | 说明 | 对策 |
| --- | --- | --- |
| 中间 chunk 被当成 decode | `preTokens != 0` 会跳过 isPrompt；`assign(1, curRet)` 清掉剩余 prompt | 显式 `prefill_remaining`；中间 chunk 不提交生成 token |
| `promptLen` 用剩余全长 | `FillLLMInputs` 的 pid = `promptLen - seqLen + i` | promptLen = 到本 chunk 末尾的绝对位置 |
| 丢掉中间 lm_head 输出 | 2500 内循环靠 `isIntermediateChunkedPrefill` 跳 head | 中间 chunk 同样设这个 flag，或接受多算一次 head |
| chunk 过小拖慢 prefill | 2048 是默认值，未必最优 | PR-B 扫 2048/4096/8192 |
| 长 prefill 与 CUDA Graph 冲突 | graph 只收 `dims[1]==1` | 不改 graph |
| page 抖动 | 多轮 `AddPrefill` | 观察 pagesLimit；峰值 KV 不变 |
| MTP 调度副本 | auto-handoff 走 `Qwen35MTPLoop` 的 `preTokens != 0` 门 | 同一套 remaining-count 接到 MTPLoop；真 MTP/DFlash 仍整段放行 |

## 7. 实施顺序

```
PR-A  完成：选批 remaining-count + 让位接到 RunNewMainLoop 与 Qwen35MTPLoop
      （FASTLLM_LONG_PREFILL_CHUNK 回滚仍在）。80K C=2 #0 产出 40；
      greedy sha256 与 chunk off 一致；8K C=4 让位 12/8/4/0。
PR-B  已关闭（死路，不是调参）。4096 已扫：80K C=2 让位 20，TTFT #1
      78.34 s，common window 77.22，sha256 仍 ea2857f4。8192 无需扫：
      chunk 8192 仍 limit==chunk（第二条 leftover 0 跳过），且
      `QWEN35_BATCH_PREFILL_SEQ_MAX=4096`（qwen3_5.cpp:477）不让
      `seqLens[b]>4096` 走 fused batch prefill（15382-15384）。

      关键：把 `GetBatchedPrefillTokenLimit()` 单独抬到 4096/8192 而 chunk
      留 2048，能让两条 2048 在 round 1 同批（线性 KV 还空，fused），但
      round 2+ 线性 KV 非空时 `canRunFusedBatchPrefill` 拒绝融合
      （15393-15401），`ForwardGPU` 退回 `runSplitBatchForward` 逐行串行
      （15303-15341，一个调度 tick 里两次 ForwardGPU(1,...)）。于是
      同一轮仍跑两条 2048 的墙钟，TTFT #0 从 41 s 回退到 ~80 s，yield
      40→0，TTFT #1 也只掉到 ~80 s。所以「抬 batch limit」≠「重叠
      prefill」，除非再写一个接受非空线性 KV 的 fused continued-prefill
      kernel（这本身是 PR-C 级别的变更）。
PR-C  重新定性（数据见 §5.1/§9）：两个候选形状都过不了剩余数字门，因为
      剩余门是 kernel 速率门，不是调度门。
      (a) fused continued-prefill（接受非空线性 KV）：总 prefill FLOPs
          不变，163840 token ÷ ~2000 tok/s ≈ 80 s——TTFT #1 最好 ~80 s
          而不是 41 s（两条仍共享同一张卡的算力），且让位→0。
      (b) mixed batch（prefill chunk + decode 同 forward）：改善 #0 的
          饿死/完成时间；TTFT #1 不变（prefill-bound），common window
          聚合不变或更低（decode 速率不动，挪到窗口外的 token 更多）。
      108.46 / 61.16 是 QPN2-on 时代的 kernel 速率；在 QPN2-off 的树上
      不可达。2026-09-15 更正：QPN2 已可缺省开，两条门均已达标
      （当前二进制 115.39 / 73.51），故 (a)/(b) 不必做；要再追就修 QPN2/N-pad
      warmup 那条线（已完成），不在本方案范围。

      (c) 前置约束（顺序安全，先于 (a)/(b) 任何实现）：本方案今天安全的
          前提是「同一条请求的 chunk 严格按序推进、不跨请求重排」。qwen3_5
          的线性注意力层带递归状态（GDN recurrentState + convCache），
          状态初始化点与回滚记账都依赖 chunk 顺序。PR-A 的 sha256 一致
          （80K C=1 `9bb0aa71`、C=2 `ea2857f4`）就是这条前提成立的证据。
          一旦 PR-C 要在同一个 forward 里交错两条请求的 chunk，或延后任一
          条请求的**首个** chunk，必须先排除需要递归初始化的首 chunk，并让
          可用窗口避开 `n_rs_seq`。参照 TurboPrefill（llama.cpp
          discussion #24092 / PR #24219）的处理：它检测 `rs_z` 后把首 ubatch
          踢出流水窗口（`turbo_start_ubatch = 1`），并把窗口收缩为
          `(n_tokens - n_rs_seq - 1) / n_ubatch`。不照抄拓扑，只照抄这个守卫。
```

PR-A 的让位门已过。B 是死路（抬 batch limit ≠ 重叠 prefill，TTFT #0 会
回退）。C 重新定性后无调度缺口：剩余数字门全是 kernel 速率门（§5.1
守恒推导 + chunk off 零让位对照 78.35）。本方案到此完整。

## 8. 取证方式

本方案的每条结论都可在本机复核：

```sh
# 请求创建时 currentTokens = 整条 prompt（分块的源）
sed -n '3277,3278p' /home/fastllm/src/models/basellm.cpp
# thisLen 的来源（选批每轮只取一个 chunk）
sed -n '2255,2258p' /home/fastllm/src/models/basellm.cpp
# 在飞分块排在新 prompt 前（两条 80K 不 ping-pong）
sed -n '1755,1758p' /home/fastllm/src/models/basellm.cpp
# compactBlock / prefillChunk 容量预留（§5.2）
sed -n '4356,4364p' /home/fastllm/src/models/basellm.cpp
# chunk 默认值（qwen3_5 = 2048）
sed -n '8839p' /home/fastllm/src/models/qwen3_5.cpp
# 分块代数与让位判据
grep -n 'LongPrefillChunkEnabled\|ShouldForceDecodeThisIteration' \
  /home/fastllm/src/models/qwen3_5.cpp /home/fastllm/src/models/basellm.cpp
```

## 8.5 C=4 与 TurboPrefill 复核（2026-09-15）

**C=4 已经能叠，靠的是池子不是守卫。** `--tokens 49152 --batch 4`（8K、
out=256）：页守卫**阻塞 0 轮**，common window **270.82 tok/s**，窗口内四条请求
各 68.97 / 68.99 / 69.13 / 69.34 tok/s，即并发步长 ≈14.5 ms = 1.26× C=1。
单请求基线 87.0，所以权重摊薄到 C=4 依然成立。C=2 同理（`--tokens 32768`
→ 窗口内 80.29 / 80.89，out=1024 时步长 13.78 ms = 1.07×）。
**结论：8K 档 C=2 用 32768、C=4 用 49152，strict 默认策略下即可，不需要改
守卫。** 49152 也是 8K C=4 不 OOM 的实用上界。

**TurboPrefill 的判定不变，且 C=4 也不改变它。** 它是 Intra-Prompt Pipeline
Scheduling，资格门要求 `split_mode == LAYER`（层划分的多卡流水线）。本机是
**TP4 张量并行**：每卡持有一层的一片、同一个 ubatch 四卡并行，**层与层之间
没有流水线可填**，机制没有宿主。它也不改变 FLOPs 或 decode 速率，所以 C=4
的 270.82 不受影响。唯一可借鉴的思想（长 prefill 分块 + 期间让位）已由 PR-A
实现，C=4 的让位读数是 12/8/4/0。**不要为它改拓扑。**

顺带否掉一条：**"chunk 按页预算收缩"（把 U2 的 (c)）实测不可行**——TTFT 能从
18.02 s 砍到 8.63 s，但逐 token 延迟从 12.62 ms 崩到 95.69 ms（7.6 倍），
地板从 2 提到 512 也没有改善，因为页需求对 chunk 长度是 128-token 的台阶函数。
接线已撤除，详见 `sm70_c2_ttft_overlap_plan.md` §4 的 U2 小节。

## 9. 未验证项

- **TurboPrefill（llama.cpp discussion #24092 / RFC PR #24219）已评估，不适用。**
  它是 Intra-Prompt Pipeline Scheduling，资格门写在 `llama-context.cpp`：只收
  `split_mode == LAYER`、多卡、非 MTP、非 embedding、因果、`n_tokens >= 2*ubatch`，
  机制是在 layer-split 的层间流水线上做「对角线波次」重放消气泡。本机是 TP4
  张量并行，每卡持有所有层的一片、同 ubatch 四卡并行，**没有层间流水线**，
  该机制无宿主。它也不改变 FLOPs 与 decode 速率，故 §5.1 的两个 kernel 速率门
  （TTFT #1 守恒、common window 108.46）不受影响。唯一可借鉴的思想（长 prefill
  分块 + 期间让位）已由 PR-A 实现。不要为它改拓扑。
- 本机只有 16GB V100。8K C=4 已测（graph-on 129.27 tok/s，让位 12/8/4/0）。
  **80K C=4 与 180K/256K 未验证**，不要外推。
- 80K C=1 graph-on prefill 1996.83 tok/s，高于方案 1971。分块后 prefill
  没有掉到 0.85 下限。
- 「先到那条在 41.6 s 内几乎零产出」已由 `request #N before last TTFT` 直接
  打印。80K C=2 改后是 40 / 0。
- `currentTokens` / `preTokens` / `promptLen` / `cacheLen` / `allTokens` 的读点
  已在 §4.1 列全。实施时按那张状态机改，不要只改 2240 的 `if`。
- 真 MTP / DFlash 长文并发未做。选批分块只开在 `mtpDraftsPerStep == 0 && !DFlash`。
- 80K C=1 graph-on decode 56.48 tok/s 对 chunk off 56.64（均为 QPN2 off）。
  61.16 是 QPN2-on 基线；原树 warmup 崩不能拿来卡 PR-A，2026-09-15 更正：
  QPN2 缺省开后实测 73.51，本门达标（当前含 combine 二进制复测值）。
- 80K C=2 graph-on common window 75.83（QPN2 off）对方案 108.46。已用同树
  chunk off 零让位对照拆账：对照 78.35（窗口 6.50 s，509 token，per-request
  39.32/39.25，TTFT 41.01/82.10，sha256 同 `ea2857f4`）。让位只值 −2.5；
  −30 来自 QPN2-off 的 batch-2 decode 速率（39.3 vs 基线 54.23）。108.46 是
  QPN2-on 时代数字，在 QPN2-off 的树上不可达，mixed batch 也回不去。窗口
  定义（`benchmark.py` `_common_decode_window`：last TTFT→batch_end）天生
  奖励零让位。**2026-09-15 当前二进制复测（QPN2 on）**：chunk off 121.90
  （窗口 82.25+4.18 s，509 token，before-last-TTFT 1）对 chunk on 115.39/115.52
  （82.87-82.95+4.07 s，470 token，让位 40），差值 −6.51；但同一对的
  `Total time` 是 86.43 s 对 86.94 s，**只 +0.7%**，TTFT min/avg/max 在 1% 内。
  即让位的真实墙钟代价 < 1%，−6.51 是窗口口径的产物。2026-09-15 更正：QPN2 缺省开后实测 115.39 且保留 40 token
  让位，本门达标（当日 16:12 在当前含 combine 二进制上复测）。
- 80K C=2 TTFT #1 守恒检查：163840 ÷ 1996.83 = 82.05 s ≈ 实测 82.81；
  基线 163840 ÷ 1958.80 = 83.65 ≈ 83.09。「明显 <83.08」⟺ prefill 吞吐
  ≫2000 tok/s（kernel 目标）。等长 prompt 下它与让位门互斥：
  `ShouldForceDecodeThisIteration` 要 decodeActive>0
  （longPrefillChunk.h:119-122），即 #0 先完成 prefill，#1 的 40 个
  chunk 只能排在其后。
- 抬 `GetBatchedPrefillTokenLimit`（chunk 留 2048）不是免费的 PR-B 赢：
  round 2+ 线性 KV 非空，`canRunFusedBatchPrefill` 拒绝融合
  （qwen3_5.cpp:15393-15401），`runSplitBatchForward` 逐行串行，TTFT #0
  回退。重叠 prefill 需要（a）接受非空线性 KV 的 fused kernel，或
  （b）PR-C mixed batch。两者都是 PR-C 级别，不是 budget 调参。
