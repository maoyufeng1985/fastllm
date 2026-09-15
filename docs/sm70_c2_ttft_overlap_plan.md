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
4. **比 +75% 更重要的副产物：87 持平表（C=1/2/4 全 87）很可能是假象。**
   那张表在 `--tokens 16384` 下测，窗口里只有单请求；"C=2 单请求 43.6"是把
   87.17 对半折算的假设值，不是实测。`t32768.out` 首次给出真正的双请求窗口：
   每请求 ~80 tok/s（并发步长 ≈ 13.3 ms ≈ 1.16× C=1），与"权重不摊销
   2.00×、每步 22.94 ms"的旧测量矛盾。二者必有一错，U0 一锤定音。
5. 执行顺序：**U0 复核实验（一锤定音）→ G1 配置落地 → U1 守卫计数器打印 →
   U2 守卫策略最小修复 → U3 让位粒度 A/B → U4 co-prefill 公平性（可选）**。

## 1. 时间线还原（t16384 vs t32768，out=32/请求）

| 阶段 | 16384（挂起） | 32768（解除） |
|---|---|---|
| #0 prefill | 0 → 3.18 s | 0 → 3.18 s |
| #0 decode | 3.18 s 起，31 token 后**退场** | 3.18 s 起，让位期只抢到 4 步 |
| #1 prefill | **被守卫挂起**，等页；3.5 s 后才跑 | 3.18 s 起，4 chunk 与 #0 的 4 步 decode 交错 |
| #1 TTFT | 6.70 s | 6.41 s |
| common window | [6.70, 7.06] s，**只有 #1**（85.83） | [6.41, 6.80] s，**两个都在**（79.61 + 80.27） |
| common window 聚合 | 85.83 tok/s | **150.19 tok/s** |

16284 那跑 #0 "消失"的原因不是异常：out=32 时 #0 在 3.5 s 就退场放页，#1 的
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

- **U0 复核实验（先做，一锤定音）**：`--tokens 32768`、8K C=2、out=1024
  （对齐旧 22.94 ms 测量的工况），各 3 次。读三样：报告逐请求 in-window
  速率；SCHED_TRACE 阻塞数（应为 0）；trace 里 C=2 并发 decode 步长。
  **门**：并发步长落在 13–15 ms（≈1.16×）或 22–24 ms（≈2.00×）二者之一，
  并与逐请求速率互洽；写进 §0.4 的裁决行。
- **G1 配置落地（与 U0 并行，零风险）**：8K 档基准命令补
  `--tokens 32768`；把 87 持平表标"16384 池、窗口单请求"的历史口径，
  以 t32768 的逐请求分解为准。
- **U1 守卫计数器**：阻塞发生时打印 `need/free/limit/请求长度/chunk` 五元组
  （SCHED_TRACE 已有轮级标志，缺账本细目）。产出：确认挂起时缺的到底是
  chunk 级还是整 prompt 级页需求。
- **U2 守卫最小修复**：按 U1 的账本选一：`(a)` 预留只计下一 chunk +
  decode 增长页，整 prompt 不预占；`(b)` chunk 尺寸自适应收缩到
  `free - decodeReserve`。env `FASTLLM_PREFILL_PAGE_POLICY=strict|grow`
  （缺省 strict=现役）。**门**：16384 池上 8K C=2 阻塞 0/72；token sha256
  与 32768 跑逐位一致；80K/180K 工况不回退、不新增 OOM。
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

- 并发 decode 步长 13.3 ms（t32768 窗口隐含）vs 22.94 ms（旧 out=1024 测量）
  的矛盾未裁决——U0。
- 守卫挂起时页账本的细目（缺的是 chunk 级还是整 prompt 级需求）未打印——U1。
- U3 让位粒度的三个档位未扫。
- U4 的 `canRunFusedBatchPrefill` 拒绝条件未梳理。
- 80K/180K 档在 G1/U2 下的回归未跑（本轮只动 8K 档）。
