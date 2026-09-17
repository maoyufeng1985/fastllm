# 2 路并发互相隐藏 + 多并发优化方案

日期：2026-09-16
对象：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2（PCIe fabric，无 NVLink）/ TP4

上游证据（本方案不重新测，只做汇总与对账）：
`docs/sm70_status_and_backlog.md` §3.1–§3.5、`docs/sm70_c2_ttft_overlap_plan.md`
（含 §0.5 的 U0 裁决）、`docs/sm70_4x200k_rotate_plan.md` §8、
`.audit/sm70-longctx-kv.tsv` 行 40–45 与 2026-09-16 各轮。

---

## 0. 结论先行

1. **这个题在本仓库已经基本答完了，但答案被两个同名的"并发"埋住了。**
   本会话反复踩的坑就是把它们当成一件事：一个是**算子级（kernel 级）并发**，
   一个是**请求级并发**。前者天花板约 3% 且实测全负；后者不但可行，
   **已经在赚**，而且是大钱。两者必须分开谈，见 §1。

2. **请求级 2 路"互相隐藏"不需要新方案，它已经实现并在跑。**
   U0 复核（3 次同 sha）：8K C=2 并发 decode 步长 **≈12.3 ms = 1.07× C=1**，
   每请求 **81.6 tok/s**、聚合 **≈163 tok/s**，而 C=1 是 87.0 tok/s。
   机制不是"两条流互相让路"，是**一份权重喂两个请求**：decode 每步要流完整整
   5.14 GB/rank 的权重，HBM 在 batch=1 时已经打满，多来一个请求几乎不额外花
   带宽。C=4 也成立：窗口内四条各 ~69 tok/s、聚合 **270.82**、步长 ≈14.5 ms
   = **1.26× C=1**。**这比算子级重叠强一个量级，而且没有死锁风险。**

3. **卡住 2 路收益的不是算力、也不是调度策略，是 KV 页池的配置。**
   `pagesLimit = totalPages * 4 / 5`（`src/models/qwen3_5.cpp:22187`）。
   8K C=2 用 `--tokens 16384` 时是 128 页、限 102 页，而两个 8K prompt 本体就要
   128 页，于是 `prefillPageCapacityBlocked = true`
   （`src/models/qwen3_5.cpp:22843`）把第 2 条的 prefill **整段挂起**，等到第 1 条
   退场放页。抬到 `--tokens 32768`（限 204）后：阻塞 **27/72 轮 → 0/45 轮**，
   common window 聚合 **85.83 → 150.19（+75%）**。**零代码。**
   页池从来不是显存限的：8K C=2 只要 +269 MB/卡，而当时 `availForKV=1.93 GB`
   能装约 918 页。

4. **4×200K 那一档的约束是准入预算，不是页池。**
   `src/models/qwen3_5.cpp:10132-10133`：开着 rotation 时一次只准入
   `2 * this->GetChunkedPrefillSize()`，因为**同一 forward 里塞 3 个及以上 chunk 会在
   批量 prefill 路径上 abort（代码注释写明是实测结论）**。所以 4 条请求是
   "两两成对"地跑：总时间 **559.77 s vs 串行 562.18 s（持平）**，TTFT 分布变成
   `[277, 277, 554, 554]`。**那一档并发只买公平性，不买吞吐**，因为总功守恒
   且设备已经 97% 忙（`docs/sm70_4x200k_rotate_plan.md` §8.3）。

5. **算子级并发（R1/R2/R5）已结案，不要再投。** 天花板是设备空闲时间
   ~3.1 s / 104.2 s ≈ **3%**；实测 R1 **−5.35%**、R2 **−7.04%**、
   R5 更慢且在 80K 把三张卡打到 `Xid 13` SM 越界。详见 §4。

---

## 1. 两种"并发"必须分开

| | 算子级（kernel 级） | 请求级（C≥2） |
|---|---|---|
| 想让什么重叠 | 同一条请求里，一个集合通信与另一个算子 | 两条请求的计算互相填补对方的等待 |
| 度量 | `max_concurrent`（单卡在飞内核数）、`t_>=2` | common window 聚合 tok/s、并发步长 / C=1 步长 |
| 现状 | **1，且 t_>=2 = 0.0000 s** | **已可用** |
| 收益机制 | 填设备空闲 | **一份权重喂 N 个请求**（摊薄） |
| 天花板 | **~3%**（稳态设备已 97.0% 忙，可填的只有 3.17 s） | decode 步长 1.07×（C=2）到 1.26×（C=4），即聚合接近线性 |
| 实测结果 | R1 −5.35%、R2 −7.04%、R5 崩 | C=2 聚合 ≈2×、C=4 聚合 **270.82** |
| 状态 | **已结案，默认全关** | **在用，继续优化** |

**为什么会混。** `max_concurrent=1` 说的是"单个设备上同时只有一条流上的内核"，
它**不**表示"机器不能同时服务两个请求"。请求级并发走的是**同一条流上更大的
batch**（M 从 1 变成 2/4），内核数照样是 1。所以拿 `max_concurrent` 去判断
"能不能做多并发"是问错了指标。

---

## 2. 2 路"互相隐藏"已经拿到什么（全部为引擎实测）

| 工况 | 读数 | 来源 |
|---|---|---|
| 8K C=1 decode | 87.0 tok/s，11.49 ms/token | `sm70_status_and_backlog.md:140` |
| 8K C=2 并发步长 | **≈12.3 ms = 1.07× C=1**（3 次同 sha `d699bcdb…`） | `sm70_c2_ttft_overlap_plan.md` §0.5 |
| 8K C=2 每请求 / 聚合 | 81.6 / **≈163 tok/s** | 同上 |
| 8K C=2（短 prompt，两请求同时在 decode） | **169.97 tok/s** vs C=1 89.22（1.91×） | `sm70_status_and_backlog.md:146` |
| 8K C=2 common window（`--tokens 32768`） | **161.72**（AR off 149.86） | `sm70_status_and_backlog.md:143` |
| 8K C=2 common window（`--tokens 16384`） | 85.83（**单请求窗口**，是口径产物） | `sm70_status_and_backlog.md:167` |
| 80K C=2 common window | **115.39**（保留 40 token 让位；零让位 121.90） | `sm70_status_and_backlog.md:145` |
| 8K C=4 common window | **270.82**，四条各 68.97/68.99/69.13/69.34，步长 ≈14.5 ms = 1.26× C=1 | `sm70_status_and_backlog.md:144` |

**两个必须小心口径的地方**（`sm70_status_and_backlog.md:200-205`）：

* **只有 `Batch decode (common window)` 能代表并发速率**，它从**最后一个** TTFT
  起算。`Batch decode after TTFT` 从**第一个** TTFT 起算，会把另一条请求还在跑的
  prefill 记进 decode，8K C=2 的 53.58 就是这么来的。
* **步长要用"每请求 token 数"去除窗口跨度**。窗口 12.53 s 里每请求各出 ~1021 个
  token，所以 12.53 / 1021 ≈ 12.3 ms；用聚合的 2042 去除会得到 6.1 ms，那是
  "每聚合 token 的时间"，不是步长。

**所以"2 路互相隐藏"的正确答案是：不要去造它，它已经是批量摊薄的副产品。
要做的是别让调度器把两条请求串起来。**

---

## 3. 收益表（剩余杠杆）

单位与工况都写在行内。**"已拿"是已经落地并进了基线的，"未做"是没有实测数字的。**

| # | 杠杆 | 预期收益（具体数字） | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|---|
| **L1** | 按工况定 `--tokens` 池（8K C=2 用 32768、C=4 用 49152；80K 档 167936） | 8K C=2 聚合 **85.83 → 150.19（+75%）**；C=4 达 **270.82** | 页池 ≥ N 个 prompt 的页需求 | +269 MB/卡（8K C=2）；抬 `--tokens` 会抬 promptLimit | **已拿**（配置，零代码；`sm70_status_and_backlog.md` §3.3） |
| **L2** | 已落地的内核收益（作为基线，不要重复计） | CUDA Graph 8K C=1 **+22.7%** / 80K C=1 **+25.7%** / 80K C=2 window **+11.5%**；QPN2 8K C=2 **+62.6%** / 80K C=2 **+48.9%** / 8K C=4 **+101.4%**；TP AR 独立 dest 8K C=2 **+7.9%**；GQA Combine 8K/80K C=1 **+10.0%** | — | — | **已拿**（`sm70_status_and_backlog.md:54-58`） |
| **L3** | **让位粒度**：让位从每 chunk（2048 token）改到每 N token（512 / 256），env 可调 | **未测**。门限已定：8K C=2 的 `Batch total` 上升、第 2 条 TTFT 劣化 ≤15%、80K C=2 不回退 | 需要改让位路径 + 扫档 | 让位太碎会让每次 forward 的固定开销压过收益（U2 就是这么崩的） | **未做**（`sm70_c2_ttft_overlap_plan.md` U3） |
| **L4** | **co-prefill**：两条请求的 chunk 放进同一批（M=4096） | **TTFT 公平性**，不是聚合：TTFT max 7.77 → ~6.4 s；聚合近中性 | 先要解除 `canRunFusedBatchPrefill` 的拒绝（`src/models/qwen3_5.cpp:15393`） | 批量 prefill 路径本身不稳（见 L5） | **未做**（可选，`sm70_c2_ttft_overlap_plan.md` U4） |
| **L5** | **放开 3+ chunk 同批**（撤掉 `2 * chunkedPrefillSize` 的准入上限） | **公平性，不是吞吐**：4×200K 从"两两成对"变成四条齐头并进，总时间仍将持平（559.77 vs 562.18 s 已证总功守恒） | 需要修批量 prefill 路径上"3 个 chunk 就 abort"的缺陷（该 abort 是实测） | 触碰一条已被证明会 abort 的路径 | **未做**（`src/models/qwen3_5.cpp:10132-10133` 注释） |
| **L6** | 80K/180K 档在 L1 口径下的回归 | **未测**。8K 档已验，80K 只验了 C=2 单点 115.39 | — | 长 prompt 的页需求更大，池的余量要重算 | **未做**（`sm70_c2_ttft_overlap_plan.md` §7 末条） |

**吞吐的真天花板要说清楚。** 4×200K 档里并发**不增加总吞吐**：总时间 559.77 s
对串行 562.18 s 持平。原因有两个，都已被实测：
① 总功守恒（同样多的 token、同样多的权重读取）；
② 稳态设备已经 **97.0% 忙**（102.10 / 105.27 s，`sm70_4x200k_rotate_plan.md` §8.3），
没有空闲可填。**所以那一档要动吞吐，只能减功（缩短上下文）或让 prefill kernel
更快，并发只能买公平性。**

---

## 4. 已否决清单（别重复投入）

| 方向 | 结论 | 证据 |
|---|---|---|
| 算子级双流重叠（R1 分块流水） | **−5.35%（80K）**；分块重叠本身 +1.58 点，但侧流地板 2.37 点 | `sm70_4x200k_rotate_plan.md` §8.2 |
| 只把 AR 放侧流（R2） | **−7.04%**；拆开是 4.37 点（去掉 host drain）+ 2.37 点（换流） | 同上 |
| 自定义 one-stage AR 走侧流（R5） | 8K **慢 13.2%**；80K **`Xid 13` SM 越界崩溃**（327 条新内核日志）。默认关，注释标禁止开启 | `sm70_4x200k_rotate_plan.md` §8.8 |
| 抬掉 SM70 force-sync 门（`FASTLLM_SM70_NCCL_ASYNC=1`） | 同步 **−45%**，但 `max_concurrent` 仍 1、墙钟 0 | `sm70_4x200k_rotate_plan.md` §8.1 |
| U2：按页预算收缩 prefill chunk，少花 269 MB 拿 L1 的收益 | **三档全废**：逐 token 延迟 **12.62 → 95.69 ms（7.6×）**；地板从 2 提到 512 token 无改善（页需求与 chunk 长度是 128-token 台阶函数）。代码已撤除 | `sm70_c2_ttft_overlap_plan.md` §U2 |
| KV 量化换速度 | **各上下文全负**：8K −1.1%、80K −4.9%、180K −8.2%。只剩省显存一个用途 | `sm70_status_and_backlog.md:189-191` |
| 调 `--tokens` 之外的守卫策略 | 8K C=4 在**默认 strict 策略**下就已经阻塞 0，不需要改守卫 | `sm70_status_and_backlog.md:144` |

---

## 5. 取证方式

```sh
# 并发基线（只信 common window）。页池必须按 N 个 prompt 定尺。
FASTLLM_SCHED_TRACE=1 python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --max_batch 4 --tokens 32768 --batch 2 --input_tokens 8192 \
  --output_tokens 32 --dtype auto --enable_thinking false --prefix_cache false \
  --temperature 0 --top_k 1 --warmup 0 2>/tmp/c2.err

# 阻塞轮数（挂起是否被解除）。0 才算 L1 到位。
grep -c "prefillBlocked=1" /tmp/c2.err

# 守卫账本（只在阻塞时打印，看出差几页）
grep -A2 "\[guard-mgr\] BLOCKED" /tmp/c2.err
```

跑多卡测试一律用看门狗包一层，它会顺手比对跑前跑后的 Xid/ECC 并抓卡死现场：

```sh
tools/gpu_watchdog.sh /tmp/c2.out --timeout 900 --label c2 -- <上面的命令>
```

---

## 6. 未验证项

* **L3 让位粒度的三档（512 / 256 / 128 token）没有扫过**，因此它的收益数字是空的。
  这是当前唯一一条"机制清楚、成本已知、数字未知"的请求级杠杆。
* **L4 的 `canRunFusedBatchPrefill` 拒绝条件没有梳理过**（`src/models/qwen3_5.cpp:15393`）。
* **L5 的 "3 个 chunk 就 abort" 只有结论没有根因**，注释里写的是实测，没写为什么。
* **80K / 180K 档在 L1 口径下的回归没跑**，本方案里 80K 只有一个 C=2 单点（115.39）。
* **`max_concurrent` 在 C≥2 下是否仍为 1** 没有量过。按 §1 的解释它应该仍是 1，
  但这只是推断；量它需要一份 nsys trace，本方案没有做。
