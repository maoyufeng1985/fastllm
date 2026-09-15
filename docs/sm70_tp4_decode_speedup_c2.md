# TP4 8K C=2 decode 加速方案

日期：2026-09-14
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP / 8K / C=2
对照：C=1 方案 `docs/sm70_tp4_decode_speedup.md`；入队分块 `docs/sm70_long_prefill_chunk_plan.md`

---

## 0. 一句话

官方 8K C=2 的 **53.58 tok/s 不是稳态 decode**。它和 80K C=2 的 6.72 / 11 tok/s
是同一个口径：窗口从第一个 TTFT 起算，把第二条 8K prefill 算进 decode。
真正要加速的是两件事，顺序不能反：

1. **让 #0 在 #1 prefill 期间继续出 token**（选批分块 PR-A）。8K 的饿死窗口约 6 s，
   不是 80K 的 41 s，但官方数字几乎全是这段。
2. **两条都进 decode 之后的 fused batch=2 图**。消息是 **20 KiB**，仍走 custom
   one-stage，不走 NCCL。杠杆仍是独立 dest 关 end barrier，不是 pack32。

不要把 C=1 的 75 tok/s 方案原样套过来，更不要把 53 tok/s 当 kernel 目标去砸。

## 1. 现在的数字怎么读

长 prompt `ftllm benchmark`（greedy，256 out，`--prefix_cache false`）：

| in | batch | TTFT min–max | Decode after TTFT | 墙钟 | 含义 |
|---|---:|---|---:|---:|---|
| 8192 | 1 | 5.92 s | **70.35 tok/s** | — | 干净 C=1 |
| 8192 | 2 | 5.94–11.87 s | **53.58 tok/s** | 15.46 s | **被 #1 prefill 污染** |
| 8192 | 4 | 5.92–23.67 s | 47.14 tok/s | 27.56 s | 同口径，更脏 |

对照已经拆开的 80K C=2：旧口径 11 tok/s，common window 108.46 tok/s，
窗口内单请求 54.23。8K C=2 **没有打过 common window**。按同样定义反推：

- #1 的 prefill ≈ 11.87 − 5.94 ≈ **5.9 s**（一条 8K prefill 的墙钟，与 C=1 TTFT 一致）
- 这段里 #0 若完全饿死，旧口径的 53 tok/s 会把这 5.9 s 和后续并发 decode 混在一起
- 墙钟 15.46 s − 共同 TTFT 起点 5.94 s ≈ 9.5 s 窗口里产出 2×255 token → 正好 ~53.6

所以 53 tok/s 的构成大概是「~6 s 零产出 + ~3.5 s 全速 C=2 decode」，
不是「decode kernel 比 C=1 慢 24%」。

短 prompt C=4（64 in）排除共同 TTFT 后是 **283.66 tok/s** / 单请求 58.5。
那才接近「并发 decode 已经在跑」的形态。8K C=2 一旦两条都进 decode，
聚合预期在 **110–130 tok/s** 量级（单请求 55–65），需要 common window 钉死。

`--max_batch 2 --batch 2` 会卡在 warmup；C=2 必须 `--max_batch 4 --batch 2`。
图预捕获默认几何 `{1, 2, 4, max}`，C=2 **在形状列表里**。

## 2. 稳态 C=2 decode 长什么样（两条都 all1 之后）

```
RunNewMainLoop 选到 2 条 preTokens>0
  seqLens = [1, 1], all1 = true, isPrefill = false
  ForwardSingleGPUDecodeGraph(batch=2)     // 资格：all1 && seqLens[b]==1
    CUDA Graph replay, 预捕获过的 batch=2 形状
    QPN2  M=2（门是 1..32，过）
    AR 消息 = 2 × 5120 × 2 B = 20 KiB
      TryTP2P2PAllReduceAdd  → TP=2 only, miss
      FastllmNcclAllReduce(hidden, hidden)   // 仍 in-place
      auto small 路径：custom one-stage + start/end barrier + fused copy-back
```

图内 20 KiB（`customAllReduceRegression`）：

| 路径 | µs |
|---:|---:|
| custom force | **21.44** |
| NCCL | 22.26 |
| graph no-end 独立 dest | **16.84** |
| graph NCCL in-place | 18.33 |
| pack32+1cat-bar（eager） | 37.09 |

auto 跟 small 走 custom，是对的。40 KiB 硬切对 C=2 用不上（C=4 才到 40 KiB）。

C=1 稳态拆解 **不能直接当 C=2 占比**。没有 fused batch=2 的干净 nsys。
能确定的只有：AR 仍是 128 次/step、每次 20 KiB、仍双 barrier；QPN2 仍覆盖
除 GDN-in 外的 208 条；attention 读 2×8K KV，不是 80K。

GDN-in → QPN2 在 C=1 上已经端到端测过：**decode +1.11%，prefill −1.43%**，
且 1/3 次 token 哈希漂移。C=2 不把它当杠杆。见 `docs/sm70_qpn_npad_design.md` §8。

## 3. 8K C=2 为什么会串行 prefill

qwen3_5 `defaultChunkedPrefillSize = 2048`。选批 `basellm.cpp:2240`：

```
if (thisLen > prefillChunkSize) {          // 8192 > 2048
    if (seqLens.size() > 0) continue;      // 本轮已有请求就拒第二条
}                                          // 第一条整段 8192 放行
```

随后 2500 行内循环把已经组好的 8192 再 Split 成 4×2048，但整段握着
`forwardLocker`。让位机制插不进这 4 个 chunk 之间，也插不进两条 8K 之间。

`interleaveActivePrefill` 对 qwen3_5 + `maxBatch>1` 默认开，8K 场景生效——
但目标对象是「一整次 forward」。一次 8K prefill 锁死 ~6 s，#0 已经 decode
也出不了 token。这就是 53 tok/s 的来源。

80K 是同一扇门放大 7 倍。PR-A 的状态机（`prefill_remaining`、中间 chunk 不
`assign(1, curRet)`）两边共用，8K 是更便宜的验收场。

## 4. 方案，按做的顺序

### M0 — 先把口径钉死（不改代码）

同一二进制、graph on、greedy、8192 in / 256 out / `--max_batch 4 --batch 2`：

1. 打 `Batch decode (common window)`（80K C=2 已经有的那行）。
   预期：窗口从 max(TTFT) 起，聚合 ≫ 53，单请求接近 C=1 的 70 的某个折扣。
2. 打 #0 在 `[TTFT_0, TTFT_1)` 的 `token_times`。预期：当前为 0。
3. 只抓 common window 的 nsys（不要再抓含第二条 prefill 的整段）。
   要看的是 `FastllmCustomAllReduceKernel` 的 grid / 耗时（20 KiB 应是
   packedCount=1280，CTA=3 不是 2）、QPN2 的 M=2 调用数、有没有 NCCL。

没有这三张表，后面的墙钟都无法归因。

过门：common window 数字进文档，再动 P0/P1。

### P0 — 选批分块（PR-A），专门打 53 tok/s 这条

规格已经写在 `docs/sm70_long_prefill_chunk_plan.md` §4。8K C=2 是它最便宜的
端到端场，比 80K 先跑：

- 每轮只取 `min(remaining, 2048)`，forward 结束还锁
- #1 的 4 个 chunk 之间插入 #0 的 decode（现成 `interleaveActivePrefill`）
- 中间 chunk 走 `prefill_remaining`，禁止 `assign(1, curRet)`

8K 验收（比 80K 合同更严、更快）：

| 项 | 门 |
|---|---|
| #0 在 #1 prefill 期间的产出 | **> 0 且持续**（当前为 0） |
| greedy sha256 vs `FASTLLM_LONG_PREFILL_CHUNK=0` | 相同（短 prompt、恰好 2048/2049/8192） |
| C=1 8K decode | ≥ 70 tok/s（不回退） |
| C=2 8K common window | ≥ 现测值（M0 钉死的那个） |
| 官方 `Decode after TTFT` | 会涨（因为饿死段变短），**不当验收主指标** |

8K 分块可能让单条 prefill 墙钟略升（4 次 launch 税）。允许 prefill tok/s 相对
整段 8192 有个小回退，换 decode 平滑。不要为了 prefill 峰值把 chunk 调回 8192。

### P1 — fused batch=2 图上的 no-end（20 KiB）

C=1 方案的 D1，但 dest 按 **20 KiB × 站点** 预分配，capture 的是 batch=2 图。

调用点不变：`qwen3_5.cpp:11324 / 12657` 的 `FastllmNcclAllReduce(hidden, hidden)`。
`addPartialToResidualReduce` 两条（attn-out、MLP-out）都要改。

探针：20 KiB no-end 16.84 vs custom-with-end 21.44（−21%），vs graph NCCL 18.33。
和 C=1 一样，**探针不是墙钟**。C=2 的 AR 观测还没有，20–69 µs 那条漂移是 C=1。

过门：

- greedy 8K C=2 sha256 与 in-place custom 相同
- common window ≥ +3%，否则 env 回滚
- batch=1 图与 batch=2 图都要注册 dest；warmup 已按几何预捕获，两张图都改

不要：把 20 KiB 改走 NCCL（图内 custom 略赢）；不要 PairAdd 合 attn+MLP
（数据依赖）；不要把 PushAdd 扩到 TP4。

### 明确不做 / 后置

| 项 | 原因 |
|---|---|
| 用 53 tok/s 当 kernel 基线 | 口径错 |
| 1Cat pack32 | 20 KiB 37 vs 21 µs |
| decode 20 KiB 换 NCCL | 图内 custom 略赢 |
| GDN-in QPN2 | C=1 已测 +1.11% / prefill −1.4%，哈希不稳 |
| QPN4 | 与 QPN2 同门；C=2 的 M=2 已在 QPN2 覆盖内 |
| XQA / E4M3 KV | 8K×2 的 attention 不是主项；80K 另测 |
| mixed batch | 非目标；PR-A 只做分块+让位 |
| 40 KiB NCCL 硬切 | C=2 用不上；C=4 才到 |

C=1 的 D1/D2 与本方案 P1 共享改图。若先做 C=1 D1，C=2 只需要确认 dest 按
`batch * hidden` 分配，不要按 10 KiB 写死。

## 5. 预期（诚实区间）

在 M0 钉死 common window 之前，墙钟只能分两段说。

**调度段（P0）**——打的是官方 53 tok/s，不是稳态：

- #0 在 #1 的 ~6 s prefill 里持续产出，速率接近当时的 C=1 decode（~70，单条占 GPU）
- 官方 `Decode after TTFT` 会明显上涨（饿死段从 6 s 变成 4 段交错）
- common window 本身几乎不变（它从最后一条 TTFT 起算）
- 这是用户能看见的「C=2 变快」，本质是延迟公平，不是 kernel

**稳态段（P1）**——两条都在 decode 之后：

- 基线未知，先用 M0 的 common window
- no-end 乐观：AR 占比若仍 ~20%，探针 −21% → 墙钟 ~4%；排队主导则 <3%，过不了门就回滚
- 聚合到不了 2×75=150：短 prompt C=4 单请求已经掉到 58。8K C=2 预期单请求
  55–65、聚合 110–130。那是 KV + M=2 GEMM 的正常折扣，不是 AR 没做好

达不到「把 53 拉到 150」。53 里有一半时间 GPU 在跑第二条 prefill。
先把那一半还给 #0，再谈 20 KiB AR。

## 6. 执行顺序

```
M0   common window + token_times + batch=2 nsys     （先于任何 PR）
P0   PR-A 选批分块，8K C=2 验收                      （与 80K 合同共用）
P1   独立 dest + no-end，dest 按 batch*hidden        （可与 C=1 D1 同一 PR）
```

P0 与 P1 正交。P0 不改 kernel；P1 不改调度。不要为了 C=2 decode 去动
`FASTLLM_PAGED_CUBLAS_CHUNK`（那是 prefill QK workspace）。

测量命令（与历史表对齐）：

```bash
PYTHONPATH=/home/fastllm/build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 4 --tokens 16384 \
  --dtype auto --enable_thinking false --prefix_cache false \
  --input_tokens 8192 --output_tokens 256 --batch 2 \
  --warmup 1 --temperature 0 --top_k 1
```

看 `Batch decode (common window)` 和每条的窗口内速率，不要看
`Batch decode after TTFT`。
