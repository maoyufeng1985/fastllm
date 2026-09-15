# 8K C=2 TTFT 不对称：根因链与优化方案

日期：2026-09-15
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2（PCIe fabric，无 NVLink）/ TP4 / no-MTP
上游：`docs/sm70_1cat_port_plan.md` §"8K 的并发聚合"（A/B 与守卫代码）、
`.audit/sm70-longctx-kv.tsv` 行 40–43（根因三轮：立→撤→直证反转）
证据：`/tmp/t16384.out`、`/tmp/t32768.out`（含逐请求窗口分解）、
`FASTLLM_SCHED_TRACE`（`qwen3_5.cpp:22888`，19 行，缺省 off）、
`qwen3_5.cpp:22067`（pagesLimit）、`qwen3_5.cpp:22713-22720`（守卫）

## 0. 结论先行

1. **你的 TTFT 观察成立，但它不是链的起点。** TTFT 3.19 / 7.77 s 的不对称，
   是 KV 页容量守卫把 #1 的 prefill 挂起造成的（`--tokens 16384` → 128 页、
   限 102 页，两个 8K prompt 需要 128 页 > 102）。直接证据：SCHED_TRACE 显示
   27/72 轮 `prefillBlocked=1 canAddPrefill=0`；因果证据：`--tokens` 翻倍后
   阻塞 27/72 → 0/45。所以优化对象是**页容量守卫**，TTFT 不对称是它的读数。
2. **87 这个数不是"两个请求各跑一半"，是"窗口里只有一个请求"。**
   `t16384.out` 逐请求分解：common window 里 request #0 产出 **0.00 tokens/s**
   （31 个 token 全部产出于 #1 TTFT 之前，随后退场），#1 独占 85.83 tok/s。
   所以 87 ≈ C=1 单请求速度，不是折半的并发速度。
3. **已验证的修法是配置级的：`--tokens` 16384 → 32768。** common window
   85.83 → **150.19 tok/s（+75%）**，窗口 31 → 58 tokens（两个请求都在），
   Batch total 9.06 → 9.42。零代码。显存代价 +269 MB/卡（128 页 × 2.10 MB），
   而 `availForKV=1.93 GB` 能装 ~918 页——**页池是配置限的，从来不是显存限的**。
4. **比 +75% 更重要的副产物：87 持平表（C=1/2/4 全 87）是假象，已被 U0 证实。**
   那张表在 `--tokens 16384` 下测，窗口里只有单请求；"C=2 单请求 43.6"是把
   87.17 对半折算的假设值，不是实测。`t32768.out` 首次给出真正的双请求窗口：
   每请求 ~80 tok/s。**U0 已裁决（见 §0.5）：并发步长 ≈12.3 ms ≈ 1.07× C=1，
   权重近似全额摊销；"不摊销 2.00× / 每步 22.94 ms"的旧测量是错的。**
5. 执行顺序：**U0 复核实验（一锤定音）→ G1 配置落地 → U1 守卫计数器打印 →
   U2 守卫策略最小修复 → U3 让位粒度 A/B → U4 co-prefill 公平性（可选）**。

## 0.5 U0 裁决行（2026-09-15，已执行）

**裁决：并发步长 ≈ 12.3 ms，即 1.07× C=1，权重近似全额摊销。** 「C>1 权重不摊销」
与「每步 22.94 ms」作废。

工况 `--tokens 32768 --input_tokens 8192 --output_tokens 1024 --batch 2`，各 3 次：

| 跑 | TPOP avg | common window | 其中 #0 / #1 | 逐请求速率 | sha256 |
|---|---:|---|---|---:|---|
| 1 | 13.78 ms | 6.35 s + 12.53 s，2042 token | 81.65 / 81.66 tok/s | ≈81.6 | `d699bcdb…` |
| 2 | 13.79 ms | 6.35 s + 12.54 s，2042 token | 81.58 / 81.59 | ≈81.6 | `d699bcdb…` |
| 3 | 13.82 ms | 6.37 s + 12.57 s，2042 token | 81.39 / 81.40 | ≈81.4 | `d699bcdb…` |

三跑同 sha、离散 0.3% 以内。读数口径要小心两处：

- **步长按"每请求 token 数"算，不是按聚合 token 数。** `Batch decode (common
  window)` 的 163 tok/s 是两请求之和；窗口 12.53 s 里每请求各出 ~1021 个 token，
  所以并发步长 = 12.53 s / 1021 ≈ **12.3 ms/步**，与 81.6 tok/s 自洽。
  用 2042 去除会得到 6.1 ms，那是"每聚合 token 的时间"，不是步长。
- **TPOP avg（13.78 ms）不是 decode 步长**，它含 prefill 相与首 token。

对照：C=1 8K 是 87.0 tok/s（11.49 ms/token）。所以 C=2 每请求 81.6 对 87.0，
单步只慢 **7%**，聚合接近翻倍。若真按 2.00× 不摊销，每步应是 ~23 ms、聚合
~43 tok/s；实测 163，差 3.7 倍。

**连带结论。** §3 的 U0 若证实 1.16× 则下调「C>1 权重不摊销是最大机会（+37%）」
并对 `sm70_concurrency_port_plan.md` 的 53.58 回填——**该条件已满足，两条都要改**。
另外 §0.3 的 +75% 口径不需要重写：它量的是页挂起解除，与步长独立。

## 1. 时间线还原（t16384 vs t32768，out=32/请求）

| 阶段 | 16384（挂起） | 32768（解除） |
|---|---|---|
| #0 prefill | 0 → 3.18 s | 0 → 3.18 s |
| #0 decode | 3.18 s 起，31 token 后**退场** | 3.18 s 起，让位期只抢到 4 步 |
| #1 prefill | **被守卫挂起**，等页；3.5 s 后才跑 | 3.18 s 起，4 chunk 与 #0 的 4 步 decode 交错 |
| #1 TTFT | 6.70 s | 6.41 s |
| common window | [6.70, 7.06] s，**只有 #1**（85.83） | [6.41, 6.80] s，**两个都在**（79.61 + 80.27） |
| common window 聚合 | 85.83 tok/s | **150.19 tok/s** |

16384 那跑 #0 "消失"的原因不是异常：out=32 时 #0 在 3.5 s 就退场放页，#1 的
prefill 这才被放行——守卫把两个生命周期彻底串行化了。

## 2. 根因链（代码位）

```
qwen3_5.cpp:22067   pagesLimit = totalPages * 4 / 5     -- 16384→128 页，限 102
qwen3_5.cpp:22713-22720
    collectPrefillPageNeeds(ctx, scheduledTokens + reserveTokens)
    if (hasPagedManagerShortage(combinedPageNeeds)) {
        prefillPageCapacityBlocked = true;   // 挂起，直到有请求退场放页
        continue;
    }
qwen3_5.cpp:22888   FASTLLM_SCHED_TRACE                  -- 每轮打印，本轮加的
```

注释写明的挂起理由：decode 请求可能暂时把空页压到太少，此时重建被逐出的长
上下文会把空页吃光、第一次 append 就失败——所以宁可等。这个保守是**对的**，
但它的粒度是"整请求等"，而不是"把 chunk 切小到当下能装下"。

## 3. 杠杆

| 杠杆 | 性质 | 预期 | 代价/风险 |
|---|---|---|---|
| **G1 `--tokens` 32768** | 配置，已验证 | common window +75%；Batch total +4% | +269 MB/卡（还有 ~1.4 GB 余量）；promptLimit 升到 26112 |
| **U2 守卫策略修复** | 代码 | 在 16384 池上解除挂起（少花 269 MB 拿到 G1 的收益） | 触碰保守守卫，须证不 OOM |
| **U3 让位粒度** | 代码/配置 | #1 prefill 期间 #0 从 4 步 → 数十步（现在接近空转） | #1 TTFT 拉长，需 A/B 定点 |
| **U4 co-prefill** | 代码 | TTFT max 7.77→~6.4 s（公平性），聚合近中性 | `canRunFusedBatchPrefill` 现拒绝融合，需另开路径 |
| U0 复核实验 | 测量 | 裁决 2.00× vs 1.16×，决定权重摊销议题的量级 | 无 |

U0 若证实并发步长 ~1.16×（权重近似摊销），则旧结论"C>1 权重不摊销是最大
机会（+37%）"要下调，`sm70_concurrency_port_plan.md` 的 53.58 解释也要回填；
若 2.00× 复现，则 150.19 的窗口另有成分，本方案 §0.3 的 +75% 口径要重写。
**这一条比其余杠杆都优先。**

## 4. 实施单元

- **U0 复核实验（已执行，2026-09-15）**：`--tokens 32768`、8K C=2、out=1024
  （对齐旧 22.94 ms 测量的工况），各 3 次。读三样：报告逐请求 in-window
  速率；SCHED_TRACE 阻塞数（应为 0）；C=2 并发 decode 步长。
  **门命中**：步长 ≈12.3 ms（**1.07× C=1**，落在 13–15 ms 一侧的下沿），
  与逐请求 81.6 tok/s、聚合 163 tok/s 三方互洽；三次 SCHED_TRACE 阻塞数
  全为 **0**，sha256 全为 `d699bcdb…`。裁决行见 §0.5。
- **G1 配置落地（已做，零风险）**：8K 档基准命令已补
  `--tokens 32768`（见 §6）；87 持平表已在 `sm70_status_and_backlog.md` §3.3
  与 `sm70_1cat_port_plan.md` §3 标注为"16384 池、窗口单请求"的历史口径，
  以 t32768 的逐请求分解为准。
- **U1 守卫计数器（已实施并实测，2026-09-15）**：`hasPagedManagerShortage` 改为
  返回阻塞的 manager（`blockingPagedManager`），诊断从**真实判定路径**长出，
  不复制判定；阻塞时在 `FASTLLM_SCHED_TRACE=1` 下打印账本，并按 manager 拆开。
  已实测（16384 池、8192×2、out=32：27 次阻塞，账本每次完全相同）：

  ```
  [guard] appendTok=2049 reserveTok=1 ctxTok=2048 preTokens=6144 pending=2048
          chunk=2049 | ownPages=48 needPages=17 accumulated=17
          free=15 maxPages=128 pagesLimit=102 pageLen=128
  [guard-mgr] BLOCKED need=17 own=17 selected=0 free=15 maxPages=128 pageLen=128
  ```

  **结论：不是"整 prompt 预占"，是差 2 页。** 三个数定住了：

  1. `ownPages=48 + needPages=17 = 65` 页，远低于 `maxPages=128`。本请求走的是
     **chunk 级增量**（`addExistingCache` 的 `totalPages - currentPages`），
     方案原先怀疑的"整 prompt 预占"**在本请求身上不成立**。
  2. 累积值 `accumulated=17` 与 `ownNeed=17` 相等，`selected=0`：**聚合里没有
     哨兵、没有别的请求的份额**。所以阻塞判据就是 `free(15) < need(17)`，
     差 **2 页**（= 2×128 = 256 token）。
  3. 2 页的缺口来自 `pagesLimit = totalPages * 4 / 5` 的 80% 保守线：16384 池
     → 128 页，限 102；8K 请求本体 64 页，两个请求 128 页 > 102。

  过程更正：中间一版账本用 `combinedNeed - ownNeed` 报 `fromOthers=2159`，
  我据此推断"H 是 INT_MAX 哨兵污染"，**那是错的**——减法在饱和值上不成立，
  与 `selected=0` 也自相矛盾。改成**直接打印** `accumulated` 后只剩 17。

  **因此 U2 的形态要重写**：不是"把整 prompt 预占改成 chunk 增量"（已经是），
  而是"差 2 页就让一整个 prefill 停摆"这个粒度。候选见 U2 小节。

- **U2 守卫最小修复（已实施并实测，结论：不可行，已撤除）**。按 (c) 做了
  "chunk 按页预算收缩"并在 16384 池 × 8K C=2 上量了三档，**三档都不能用**：

  | 配置 | 阻塞轮 | TTFT max | TPOP avg | common window |
  |---|---:|---:|---:|---:|
  | strict（现役） | 27 | 18.02 s | **12.62 ms** | 86.98 |
  | grow，地板 2 token | **0** | 8.63 s | **95.69 ms** | 69.65 |
  | grow，地板 512 token | **137** | 8.64 s | **95.93 ms** | 69.62 |

  三条读数定死这件事：

  1. **TTFT 确实砍半**（18.02 → 8.63 s），说明"少挂起就能更早开始 prefill"
     这个方向没错；Total time 也从 29.78 s 降到 21.69 s。
  2. **但逐 token 延迟崩了 7.6 倍**（12.62 → 95.69 ms/token，max 173 ms）。
     分块变小以后，一条 1024-token 的 prefill 要拆成大量小 forward，每次
     forward 的固定开销（调度器迭代、AR、图启动）压过了收益。
  3. **地板从 2 提到 512 没有任何改善**（95.93 vs 95.69 ms，阻塞反而 0→137），
     说明收缩总是撞地板——页需求与 chunk 长度是**台阶函数**（一页 128 token），
     一旦当前 chunk 需要 17 页而只剩 15 页，缩到 1 token 也还是需要 16 页。
     **这条策略没有可用的甜点区。**

  `grow` 的 sha 与 strict 不一致（`84e34aed…` vs `25ff2ddb…`），但它生成的目标
  token 数也不同（1040 vs 2048），属于口径差而非数值漂移——不影响上面的结论，
  因为延迟崩坏本身就足以否决。

  **处置**：接线、`FASTLLM_PREFILL_PAGE_POLICY` 开关、以及 `longPrefillChunk.h`
  里的 `SelectFittablePrefillChunkLen` 已全部撤除（不留 `[废]` 代码）。U1 的
  账本与 `blockingPagedManager` 重构保留——它们零开销且下次定位还要用。

- **C=2 / C=4 的正确杠杆是池子，不是守卫**（2026-09-15 实测）：

  | 工况 | 池 | 阻塞轮 | common window | 窗口内逐请求 | 步长 |
  |---|---:|---:|---:|---|---:|
  | 8K C=2 out=32 | 16384 | 27 | 86.18 | 只有 #1 | — |
  | 8K C=2 out=32 | **32768** | **0** | **151.34** | 80.29 / 80.89 | ~12.4 ms |
  | 8K C=2 out=1024 | 32768 | 0 | — | 81.65 / 81.66 | **13.78 ms** |
  | 8K C=4 out=256 | **49152** | **0** | **270.82** | 68.97 / 68.99 / 69.13 / 69.34 | ~14.5 ms |

  读法：C=4 在默认 strict 策略下就已经全通（阻塞 0），四条请求在窗口里各拿
  ~69 tok/s，聚合 270.82；单请求基线 87.0，所以**权重摊薄到 C=4 仍然成立**
  （C=4 每请求 69 对 C=1 的 87，单步慢 26%）。C=2 同理（81.6 对 87）。
  **所以不需要改守卫，改 `--tokens` 就够**：8K C=2 用 32768、C=4 用 49152。
  `49152` 也是 8K C=4 不 OOM 的实用上界（见整合稿 §6 风险行）。

- **U3 让位粒度**：PR-A 的让位从每 chunk（2048 token）改为每 N token
  （512 / 256），env 可调。**门**：8K C=2 Batch total 提升；#1 TTFT 劣化
  ≤15%；80K C=2 不回退。
- **U4 co-prefill（可选，另立验收）**：两请求 chunk 同批（M=4096）。
  收益是 TTFT 公平性不是聚合；需先解除 `canRunFusedBatchPrefill` 拒绝。

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| U2 放松守卫后，decode 把页吃光、prefill append 失败（守卫注释里的原始担忧） | 保留 decode 侧 `collectDecodePageNeeds` 检查不动；U2 的预留下限 = decode 增长页；失败路径回 `strict` |
| 87 持平表若被 U0 证实为真（2.00×），本方案 +75% 的口径仍成立（页挂起独立于步长），但 §0.4 要改写 | U0 先行 |
| `--tokens` 提档改变 promptLimit，影响长 prompt 工况的守门 | 各工况本就各用各的 `--tokens`（80K 档 167936）；8K 档 32768 的 promptLimit 26112 > 8192×2，无回退 |
| SCHED_TRACE 遗留开销 | 已是 env 门控、缺省 off、只读打印（19 行） |

## 6. 取证方式

```sh
# A/B（已跑，报告在 /tmp/t16384.out /tmp/t32768.out）
FASTLLM_SCHED_TRACE=1 python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 4 --tokens 16384 --dtype auto \
  --enable_thinking false --prefix_cache false --input_tokens 8192 \
  --output_tokens 32 --batch 2 --warmup 0 --temperature 0 --top_k 1 2>/tmp/t16384.err
# U0：同上，--tokens 32768 --output_tokens 1024，trace 另抓 nsys 一份
# 阻塞计数：grep -c prefillBlocked=1 /tmp/t*.err
```

## 7. 未验证项

- ~~并发 decode 步长 13.3 ms vs 22.94 ms 未裁决~~ —— **U0 已裁决**：
  ≈12.3 ms（1.07×），22.94 ms 作废，见 §0.5。
- 守卫挂起时页账本的细目（缺的是 chunk 级还是整 prompt 级需求）未打印——U1。
- U3 让位粒度的三个档位未扫。
- U4 的 `canRunFusedBatchPrefill` 拒绝条件未梳理。
- 80K/180K 档在 G1/U2 下的回归未跑（本轮只动 8K 档）。
