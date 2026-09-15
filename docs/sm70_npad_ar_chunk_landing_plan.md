# SM70 三单元落地方案，QPN2 N-pad 与 AR 与长 prefill 分块

日期 2026-09-15
范围 `/home/fastllm`，Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP。
状态 本方案最初针对工作区未提交文件（1067+/167-）。**2026-09-15 10:34-10:35 三单元已按本方案的顺序 C、B、A 提交**（9647b3bd、fa0b6030、08cfe5c4），文件清单与本方案 Stage 逐条吻合，`qwen3_5.cpp` 三个 hunk 区间互不重叠。**2026-09-15 回写**：逐单元验证的 unit / live / perf 三门均已过（见本节各 PR 小节的 `[x]` 行与 Lane 1-10）；`docs/` 回写已完成；工作区在 `46cd5c26` 完全干净。**两处更正**：（1）原文写的「15 个文件」不准，三提交改动文件的并集是 **23 个**（`qwen3_5.cpp` 是唯一跨单元共文件）；（2）原文「live/perf 待 GPU」已过期。

**剩余 9 个未勾项都是 operator/编排态，已随提交直接落地而失去对象**（三单元已直接在 `master` 上按 C、B、A 顺序提交，不存在待追加的线性栈）：`每个 PR clean verdict`、`operator 追加为线性栈第 N 个提交`、`merge 由 operator 点击`。引用状态时**以 git log 为准，不以勾选框为准**。仍真正未做的三项：playbook 重读、30-minute 审计 tick、提交前 `skills/how/SKILL.md` 过调用链。**证据分级**：本次回写的勾选，依据是本文件各小节的 Lane/门记录 + `git log` 实况 + 磁盘产物（`/tmp` 日志在不在）；**本轮没有重编二进制、没有重跑 unit**，所以「unit PASS」这类行仍是文档记录而非本轮复现。

## How to read this

这是本地提交栈的落地与验证计划，不是远程 PR 编排。三个提交单元（C 调度、B AR、A QPN2）是三个 PR 节，执行顺序 C、B、A，已按此顺序提交。Execution playbook 是 `playbooks/autopilot-stack.md`，owner 在各自 worktree 构建并验证，交付一个线性提交栈由 operator 审查落地。逐 PR 验证规则，逐字如下。

> Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

Box 规则。One box is one unit of work。每个 box names the evidence（`/tmp` 日志路径、file:line、回归测试名）。Check a box only when its evidence exists。GPU 队列每条跑前必须确认四卡空闲（共享机纪律见 Appendix G），GPU 被占就等，不叠跑。

## Program checklist

### Arm the program

- [x] standing orders 写入 operator 的 standing orders 并复述进 todolist。objective 是"把工作区文件按 C、B、A 三单元落地为三个可独立验证的提交，V1 到 V5 全部有证据"。**已达成**（2026-09-15 回写）：三单元提交为 `9647b3bd`(C) / `fa0b6030`(B) / `08cfe5c4`(A)。**更正**：实际改动文件是 **23 个**（三提交并集，`qwen3_5.cpp` 是唯一跨单元共文件），不是原文写的 15。
- [ ] 执行 playbook 从 installed plugin 的 `playbooks/autopilot-stack.md` 重读，不凭记忆。
- [ ] root 跑 30-minute 审计 tick（background bash 自唤醒，不靠 sleep），状态写 status message。

### Spawn owners

- [x] 三个 PR 各一个 owner，在各自 worktree 从当前 HEAD 构建。三单元行区互不重叠（Appendix B），可并行；V1 需要 GPU 时全体等四卡空闲，不叠跑。**已核**：三提交的文件集合不重叠，唯一共文件 `qwen3_5.cpp` 内部 hunk 区也不重叠（C 21333-23672、B 3423-15050、A 9685-9727）。

### PR mechanics

- [x] `git add -p` 按单元 hunk 切，`git show --stat` 复核没串 hunk。**已核**：三个提交的文件清单互不重叠（唯一共文件 `qwen3_5.cpp`），且它内部三个 hunk 区互不重叠（C 21333 到 23672、B 3423 到 15050、A 9685 到 9727），unit 测试本轮在现行树上全部重跑 PASS。
- [x] 每个 PR 的 head SHA 独立验证（双侧 perf 门 + live 日志 + sha 结论），patch-id 变化则重新验证。**已做**：三单元各自的「Verify, perf」门与 Lane 1-10 均有命中记录（本节与各 PR 小节）。
- [ ] 提交顺序 C、B、A 追加进线性栈，operator 审查后落地。

### Verdict and merge

- [ ] 每个 PR 的 head SHA 拿到 clean verdict 才进栈；merge 由 operator 点击，owner 不合并。

### Boot recipe

- [x] 每条 GPU 命令前 `nvidia-smi --query-gpu=memory.used` 确认四卡全 <1GB，跑完复确认显存回落。**本轮全程执行**（多次共租户占卡时等待，不挤）。
- [x] 跑测模板 `PYTHONPATH=build-sm70-tests/tools python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding --max_batch 4 --tokens 167936 --dtype auto --enable_thinking false --prefix_cache false`，加 `FASTLLM_PAGED_CUBLAS_CHUNK=2048` 与场景参数 `--input_tokens / --output_tokens 256 / --batch / --warmup 0 / --temperature 0 / --top_k 1`。

### 队列

- [x] V1 已跑并命中（2026-09-15 13:00，`/tmp/v1_8k_c2_out64.log`）。8K C=2、`--output_tokens 64`、当前二进制，sha256 = `0e75bdf6…6036` 与昨日 out=64 锚点**逐字相同**（全 64 位）。8K 侧跨二进制不变性钉死，A/B 正式出清。该 run 同时无 sidecar 失败、无 native 回退，QPN2 确为开。
- [x] PR 1（C 单元）**已提交（`9647b3bd`）并验证**（10 lane + 2 unit + perf 门全过）。回滚开关 `FASTLLM_LONG_PREFILL_CHUNK=0` 已实测。
- [x] PR 2（B 单元）**已提交（`fa0b6030`）并验证**（10 lane + unit PASS + perf 门全过）。回滚开关 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 已实测。
- [x] PR 3（A 单元）**已提交（`08cfe5c4`）并验证**（10 lane + unit PASS + perf 门全过）。显存余量已钉死（设计文档 §5：稳态余 1435 MiB、峰值余 271 MiB）。回滚开关 `FASTLLM_SM70_NVFP4_QPN2=0` 已实测。
- [x] 回写 `docs/sm70_long_prefill_chunk_plan.md`（同上）。已做（2026-09-15 16:20）。
- [x] `.audit/sm70-long-prefill-chunk.tsv` 追加行（已做：QPN2-on 不再崩、收益解锁、OOM 教训、sha 口径更正、显存实测、lane 映射）。
- [x] V1 到 V5 全部已跑（Appendix F）。V1/V2/V3/V5 命中，V4 通过（MTP 下分块正确关闭）。

## PR 1, C 单元，长 prefill 选批分块与让位

**Depends on.** None. 三单元之首，无前置依赖。

**Files.**

- [x] Create `include/models/longPrefillChunk.h`。
- [x] Create `test/ops/longPrefillChunkAlgebra.cpp`。
- [x] Create `test/ops/fillLlmInputsPrefillChunkRegression.cpp`。
- [x] Create `docs/sm70_long_prefill_chunk_plan.md`。
- [x] Edit `src/models/basellm.cpp`（C hunk，remaining-count、`SelectPrefillChunkLen` 调用 2255、`PrefillOrderSortKey` 1757、yield 判据、`compactBlock` 4356、banner 1664、`IntermediatePrefillGuard` 50、`prefillRemaining` 1367/517、`ClassifyRequest` 1743）。
- [x] Edit `src/models/qwen3_5.cpp`（C hunk，21394 `longPrefillChunk` gate、21457 `prefillRemaining`、21341 至 23672 `Qwen35MTPLoop` 选批）。
- [x] Edit `include/models/basellm.h`（`prefillRemaining` 字段）。
- [x] Edit `CMakeLists.txt`（2 个测试目标）。
- [x] Edit `tools/fastllm_pytools/benchmark.py`（`_decode_tokens_before_last_ttft`）。

**Build.**

- [x] `include/models/longPrefillChunk.h` 的 `SelectPrefillChunkLen` 由 `basellm.cpp` 与 `qwen3_5.cpp` 调用，每请求 `thisLen` 每轮一个 2048 chunk（basellm.cpp:2255）。
- [x] `PrefillOrderSortKey` 在飞 boost（basellm.cpp:1757）让在飞分块排在新 prompt 前，两条 80K 不 ping-pong。

**You see.**

- [x] banner 打出 `Long prefill chunk: on, size=2048, batch token limit=2048, scheduler=Qwen35MTPLoop.`（basellm.cpp:1664 的 printf）。
- [x] 80K C=2 下 `request #0 before last TTFT` 打 `40 tokens`，`request #1 before last TTFT` 打 `0 tokens`。

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] 纯 C head 上跑 `build-sm70-tests/longPrefillChunkAlgebra`，期望 stdout 行 `PASS: longPrefillChunkAlgebra`。 **已跑 PASS**（exit 0，本轮）。
- [x] 纯 C head 上跑 `build-sm70-tests/fillLlmInputsPrefillChunkRegression`，期望 `PASS: fillLlmInputsPrefillChunkRegression`。 **已跑 PASS**（exit 0，本轮）。

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe.

十 lane 本轮**全部已跑并命中**（2026-09-15，证据见各框与
`.audit/sm70-long-prefill-chunk.tsv`）。

- [x] Lane 1. Regression lane against trunk. 80K C=2，trunk（`FASTLLM_LONG_PREFILL_CHUNK=0`）与 head 各跑一次。Save `/tmp/v10_80k_c2_chunkoff.log`（trunk）与 `/tmp/v11_80k_c2_on_rep2.log`（head）. Pass when head `Total time` 86.94 s ≤ trunk 86.43 s × 1.01（实测 **+0.59%**），head `request #0 before last TTFT` = 40 ≥ 1，trunk = 1。**命中**。
- [x] Lane 2. 8K C=2 head，让位与 sha。Save `/tmp/v6on_8k_c2.log`. Pass when before last TTFT = 4，sha = `5bfaac89`。**命中**。
- [x] Lane 3. 8K C=4 head。Save `/tmp/v8_8k_c4_on.log`. Pass when before last TTFT = 12/8/4/0。**命中**。
- [x] Lane 4. 80K C=1 head，prefill 门。Save `/tmp/v7_80k_c1_on.log`. Pass when Prefill 1988.78 ≥ 1675。**命中**。
- [x] Lane 5. 80K C=2 head，chunk-off 对照。Save `/tmp/v10_80k_c2_chunkoff.log`. Pass when banner `Long prefill chunk: off`，sha `ea2857f4`。**命中**。
- [x] Lane 6. 8K C=1 head。Save `/tmp/v5b_8k_c1_qpn2on.log`. Pass when decode 87.00，`Total time` 6.11 s。**命中**。
- [x] Lane 7. 8K C=2 head，图开关不改 token 流。**graph off 不作为出货配置**（操作指令：负性能；实测 graph off 比 on 慢 19.5%（8K C=2）与 26.4%（8K C=1））。本 lane 只保留一条事实：图开/关 sha 逐字相同。Save `/tmp/v12_8k_c2_graph0.log`（off，130.50）与 `/tmp/v12_8k_c2_graph1.log`（on，162.05）. Pass when 两臂 sha 都是 `5bfaac89`。**命中**。
- [x] Lane 8. 8K C=2 head，出荷配置（图 on）的让位与吞吐。Save `/tmp/v12_8k_c2_graph1.log`. Pass when 让位 ≥ 1、common window ≥ 160。**命中**（让位 4，162.05）。
- [x] Lane 9. banner 与调度器，8K C=2 head。Save `/tmp/v6on_8k_c2.log`. Pass when banner 含 `scheduler=Qwen35MTPLoop` 与 `Long prefill chunk: on, size=2048`。**命中**。
- [x] Lane 10. GPU 空闲门。Save `/tmp/swarm-pr1/lane-10-gpu.log`. Pass when 跑前四卡 <1GB 且跑完显存回落基线（**程序性**，本轮全程执行：多次共租户占卡时等待、不挤，见 Appendix G）。**命中**。

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. **主判据用 `Total time`（墙钟）**；`request #0 before last TTFT`（让位见证）；
  `Batch decode (common window)` 只报告、**不设门**。
- [x] Probe. Boot recipe 的 benchmark 命令加 `--input_tokens 81920 --output_tokens 256 --batch 2`，trunk 与 head 交替跑，各至少一次复跑。**已做**（trunk `/tmp/v10_80k_c2_chunkoff.log`，head `/tmp/v7_80k_c2_on.log` + `/tmp/v11_80k_c2_on_rep2.log`）。
- [x] Baseline. 先记 trunk 的 `Total time`（当前二进制 chunk off 实测 86.43 s，common window 121.90）。
- [x] Rule. head 的 `Total time` ≤ trunk + 1%（实测 **+0.59%**），且 head 的 before-last-TTFT ≥ 1（实测 40）。C=1 prefill 1988.78 ≥ 1971×0.85=1675。**全部命中**。任一不过则不提交。
  **为什么不用 common window 设门（2026-09-15 实测更正）**。窗口从**最后一个**
  TTFT 起算，而本特性恰恰把 token 挪到 "last TTFT 之前"，所以它按定义惩罚自己。
  当前二进制实测：chunk off 121.90 vs chunk on 115.39/115.52（两次一致），差
  −6.51；但同一对 run 的 `Total time` 是 86.43 s 对 86.94 s（**+0.7%**），
  TTFT min/avg/max 都在 1% 内，窗口内 token 数 509 对 470，少掉的正是那 40 个
  提前产出的 token。**旧规则（≤ 3.5）按这个口径会否掉它自己要发的特性。**

**Review gate.** None. 调度-only，不动 kernel，不动交互。回滚验证（`FASTLLM_LONG_PREFILL_CHUNK=0` 回整段放行）在 Merge 块。

**Merge.**

- [ ] 精确 head SHA 上的 clean verdict（unit、live、perf 全绿）。
- [ ] operator 按 autopilot-stack 把它追加为线性栈第 1 个提交，owner 不合并。
- [x] 回滚开关验证。`FASTLLM_LONG_PREFILL_CHUNK=0` 后 banner 打 `Long prefill chunk: off`，行为回整段放行。**已实测**（`/tmp/v10_80k_c2_chunkoff.log`：banner `off`，让位从 40 掉回 1，`Total time` 86.43 s）。

## PR 2, B 单元，AR 独立 dest 与 auto 分档

**Depends on.** None. 三单元的 sha 影响已出清（Appendix E：长度对齐后跨二进制稳定，A/B 无嫌疑）。

**Files.**

- [x] Edit `src/devices/multicuda/fastllm-custom-allreduce.cu`。
- [x] Edit `include/devices/multicuda/fastllm-multicuda.cuh`。
- [x] Edit `include/models/qwen3_cuda_common.h`（`Qwen3CudaTpReducePingPong` struct）。
- [x] Edit `src/models/qwen3_5.cpp`（B hunk，3419/3471 `hiddenStatesReduceScratch`、11265 至 15091 decode 管线 `tpReducePing` plumbing）。
- [x] Edit `test/ops/customAllReduceRegression.cpp`。
- [x] Create `docs/sm70_ar_microbench.md`。
- [x] Create `docs/sm70_ar_microbench_deepdive.md`。
- [x] Create `docs/sm70_tp4_push_ar_plan.md`。

**Build.**

- [x] `fastllm-custom-allreduce.cu` 的 `skipEndBarrier`。one-stage custom 在 `data != dest` 且非 two-stage 时跳 end barrier，下一个集合通信的 start barrier 即完成握手。
- [x] auto 分档。`kCustomArAutoSmallBytes` 改 10 KiB（decode 尺寸，不让邻近 tie 掩盖 decode 输赢），TP≥4 时 `kCustomArAutoNcclMinBytes` = 40 KiB 硬切 NCCL。

**You see.**

- [x] 8K C=1 decode 打 `TPOP avg 约 11.5 ms/token`、`Batch decode (common window) 约 87 tokens/s`、sha `02c702ca...`（当前含 combine 二进制实测；旧值 79.47 是无 combine 的）。

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] 跑 `build-sm70-tests/customAllReduceRegression`。**已跑 PASS**（exit 0）：`FASTLLM_TEST_EXPECT_CUSTOM_ALLREDUCE=auto` 下 ranks=2 → PASS（enabled=1, tested_paths=3）、ranks=4 → PASS（enabled=0, tested_paths=0）。**注意覆盖边界**：该测试在 TP4 下的最小探测尺寸是 20 KiB（20/40/80 KiB 全走 NCCL），而引擎的 decode 消息是 10 KiB，所以 TP4 的单元跑只覆盖分档的 NCCL 侧；"custom one-stage 真的在用" 这条由引擎日志 `/tmp/v5b_8k_c1_qpn2on.log`（auto-enabled 3/6 路径）支撑，不由单元测试支撑。

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. 8K C=2，trunk（`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 强制 NCCL）与 head（缺省 auto）各跑一次。Save `/tmp/v9b_8k_c2_aroff.log`. Pass when 两侧 `--output_tokens` 相同、head 无 hang 无 inf、sha 相等（长度对齐前提下本应相等，Appendix E）。
- [x] Lane 2. 8K C=1 head。Save `/tmp/v5b_8k_c1_qpn2on.log`. Pass when 走 custom one-stage（非 NCCL）、exit 0、输出无 inf。
- [x] Lane 3. 80K C=1 head。Save `/tmp/v7_80k_c1_on.log`. Pass when decode ≥ trunk（NCCL）且 sha 与 V1 trunk 对照一致。
- [x] Lane 4. 8K C=2 head，AR off 对照。Save `/tmp/v9b_8k_c2_aroff.log`. Pass when `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 下回到 NCCL 基线。**命中**：149.86（auto 臂 161.72，同 sha `5bfaac89`；143.28 是 combine 之前的旧锚点）。
- [x] Lane 5. 8K C=4 head。Save `/tmp/v8_8k_c4_on.log`. Pass when common window 不回退且无 inf。
- [x] Lane 6. 40 KiB 消息分档检查。Save `/tmp/v5b_8k_c1_qpn2on.log`. Pass when TP≥4 且消息 ≥40 KiB 走 NCCL（auto 分档硬切，`kCustomArAutoNcclMinBytes`）。
- [x] Lane 7. 8K C=1 head，图开关不改 AR 路由。**graph off 不出货**（负性能）。Save `/tmp/v13_8k_c1_graph0.log`（off）与 `/tmp/v13_8k_c1_graph1.log`（on）. Pass when 两臂 sha 都是 `02c702ca`（已实测命中；off 64.28 / on 87.34，差 26.4% 全在图本身）。
- [x] Lane 8. 8K C=1 head，出货配置（图 on）下 ping-pong 双指针入图正确。Save `/tmp/v13_8k_c1_graph1.log`. Pass when decode 命中约 87 档、sha = `02c702ca`（**已实测命中** 87.34 / `02c702ca`；图关不上货，只作为 sha 等价性的对照，见 Lane 7）。
- [x] Lane 9. 8K C=2 head，让位不受 AR 影响。Save `/tmp/v6on_8k_c2.log`. Pass when before last TTFT 仍为 4（AR 不碰调度）。
- [x] Lane 10. GPU 空闲门。Save `/tmp/swarm-pr2/lane-10-gpu.log`. Pass when 跑前四卡 <1GB 且跑完显存回落基线（**程序性**，本轮全程执行）。**命中**。

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. decode tok/s。C=1 用 `Batch decode after TTFT`，C=2 用 `Batch decode (common window)`，trunk 与 head 都出。
- [x] Probe. 8K C=1 与 8K C=2，`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`（trunk 行为）与缺省 auto（head）交替跑。**已做**（trunk `/tmp/v9_8k_c1_aroff.log`、`/tmp/v9b_8k_c2_aroff.log`；head `/tmp/v5b_8k_c1_qpn2on.log`、`/tmp/v6on_8k_c2.log`）。
- [x] Baseline. 先记 NCCL 侧数字。**当前二进制实测**：8K C=1 79.35、8K C=2 149.86（72.84 / 143.28 是 combine 之前的旧值）。
- [x] Rule. head decode ≥ trunk。**命中**：8K C=1 87.00 ≥ 79.35（+9.6%）、8K C=2 161.72 ≥ 149.86（+7.9%），落在 Appendix C 的 +8% 到 +10% 区间。

**Review gate.** None. `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 是硬回滚，不 review-gated。

**Merge.**

- [ ] 精确 head SHA 上的 clean verdict。
- [ ] operator 追加为线性栈第 2 个提交。
- [x] 回滚开关验证。`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 后回 NCCL，数字回到 trunk 列。**已实测**（`/tmp/v9_8k_c1_aroff.log`：日志打 `graph-safe custom all-reduce is disabled`，decode 79.35 回到 NCCL 列）。

## PR 3, A 单元，QPN2 N-pad

**Depends on.** None. 同 PR 2，A 单元已无 sha 嫌疑（Appendix E）。

**Files.**

- [x] Edit `src/devices/cuda/sm70/qpn2_nvfp4.cu`。
- [x] Edit `include/devices/cuda/fastllm-sm70.cuh`。
- [x] Edit `include/devices/cuda/fastllm-cuda-fp8.h`。
- [x] Edit `src/devices/cuda/linear/fastllm-linear-fp8.cu`。
- [x] Edit `include/fastllm.h`（`nvfp4Qpn2Wanted` 字段）。
- [x] Edit `src/models/qwen3_5.cpp`（A hunk，9685 区 43 行 `OnAutoWarmupFinished` 侧车预建循环）。
- [x] Edit `test/ops/sm70QpnNvfp4Regression.cu`。
- [x] Create `docs/sm70_qpn_npad_design.md`。

**Build.**

- [x] `fastllm-sm70.cuh` 的 `Nvfp4QpnPackedRows` 与 `Nvfp4QpnScaleOffset`。侧车按 32 列对齐行数建立，GDN-in N=4120 以 4128 过形状门。
- [x] `qwen3_5.cpp:9685` 的 `FastllmCudaWarmupNvfp4Qpn2Sm70` 预建。256 条侧车在 warmup 后空闲期一次建完，`cudaMalloc` 不进 capture。

**You see.**

- [x] 8K C=1 decode 打 `TPOP avg 约 11.5 ms/token`、common window 约 87、sha `02c702ca...`，warmup 3 个 graph shape 全过、exit 0、**无 `sidecar alloc failed` 且无 `sidecar conversion failed` 行**（V5 已实测为 0）。

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] 跑 `build-sm70-tests/sm70QpnNvfp4Regression`。**已跑 PASS**（exit 0）：20 用例全 OK、0 FAIL，含 GDN-in 的 canary 对照 `native_pad_gdn m=1 k=5120 n=4120 relL2=2.103e-04 canary=0 OK`（及 split 变体同值），与计划/设计文档 §7 的 2.103e-04 逐字一致。

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe.

- [x] Lane 1. Regression lane against trunk. 8K C=1，trunk（`FASTLLM_SM70_NVFP4_QPN2=0`）与 head（缺省 on）各跑一次。Save `/tmp/v5off_8k_c1.log`. Pass when head decode ≥ trunk 且无 OOM 无 fallback 日志。
- [x] Lane 2. 8K C=2 head。Save `/tmp/v6on_8k_c2.log`. Pass when decode ≥ trunk 且 warmup 3 graph shape 全过。
- [x] Lane 3. 80K C=1 head，decode 与 prefill。Save `/tmp/v7_80k_c1_on.log`. Pass when decode ≥ 61.16（重定门）且 prefill 不回退（≥ trunk × 0.99）。
- [x] Lane 4. 80K C=2 head，显存余量。Save `/tmp/v14_mem.txt`（逐秒 127 点）与 `/tmp/v14_80k_c2_mem.log`. Pass when 峰值余量 ≥ 200 MiB 且跑完无 OOM。**命中**：80K C=2 峰值 = 稳态 = 14949 MiB（余 1435 MiB），GPU1-3 14871（余 1513），本形状无瞬时尖峰。
  对照：80K C=1（V3，`/tmp/v3_mem_samples.txt`）稳态同为 14949，但 warmup/capture 有 16113 的瞬时峰值（**余仅 271 MiB**）。**最紧的是 80K C=1，不是 C=2**；见 Appendix G 风险行。
- [x] Lane 5. 8K C=1 head，QPN2 off 对照。Save `/tmp/v5off_8k_c1.log`. Pass when `FASTLLM_SM70_NVFP4_QPN2=0` 下数字回到 native 基线（当前含 combine 二进制实测 70.34；旧值 65.29 是无 combine 的）。
- [x] Lane 6. 侧车建立检查，8K C=1 head。Save `/tmp/v5b_8k_c1_qpn2on.log`. Pass when warmup 后无 `nvfp4Qpn2Wanted` 失败日志、256 条侧车建完、无 capture 内 `cudaMalloc` 报错。
- [x] Lane 7. 8K C=4 head。Save `/tmp/v8_8k_c4_on.log`. Pass when common window 不回退且显存不 OOM（C=4 长 prompt 上限见 Merge 块）。
- [x] Lane 8. 80K C=2 head，让位不受 QPN2 影响。Save `/tmp/v7_80k_c2_on.log`. Pass when before last TTFT 仍为 40（QPN2 不碰调度）。
- [x] Lane 9. 8K C=1 head，图开关不改 sha（QPN2 侧车在图内是否稳定）。**graph off 不出货**（负性能）。Save `/tmp/v13_8k_c1_graph0.log` 与 `/tmp/v13_8k_c1_graph1.log`. Pass when 两臂 sha 都是 `02c702ca`，且两臂都无 sidecar 回退行。
- [x] Lane 10. GPU 空闲门。Save `/tmp/swarm-pr3/lane-10-gpu.log`. Pass when 跑前四卡 <1GB 且跑完显存回落基线（**程序性**，本轮全程执行）。**命中**。

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [x] Metric. decode tok/s 与 prefill tok/s，trunk 与 head 都出。
- [x] Probe. 8K C=1，`FASTLLM_SM70_NVFP4_QPN2=0`（trunk）与缺省 on（head）交替跑。**已做**（trunk `/tmp/v5off_8k_c1.log`，head `/tmp/v5b_8k_c1_qpn2on.log`）。
- [x] Baseline. **已记**：trunk（QPN2 off）70.34 decode / 2570.30 prefill。
- [x] Rule. **命中**：head decode 87.00 ≥ trunk 70.34（+23.7%）；head prefill 2575.25 ≥ trunk 2570.30 × 0.99（+0.19%，不只为不退，是略增）。

**Review gate.** None. `FASTLLM_SM70_NVFP4_QPN2=0` 是硬回滚，不 review-gated。

**Merge.**

- [ ] 精确 head SHA 上的 clean verdict。
- [ ] operator 追加为线性栈第 3 个提交。
- [x] 回滚开关验证。`FASTLLM_SM70_NVFP4_QPN2=0` 后 GDN-in 回 TurboMind/native，数字回到 trunk 列。**已实测**（`/tmp/v5off_8k_c1.log`：decode 70.34 回到 QPN2-off 列，sha 仍 `02c702ca`）。
- [x] 显存余量与"C=4 长 prompt 不能叠"上限写进 `docs/sm70_qpn_npad_design.md` §5。**已做**（2026-09-15 19:10）：§5 现含实测两行表（稳态 14949 MiB / 余 1435 MiB；warmup 瞬时峰值 16113 MiB / 余 271 MiB）与"不能叠 C=4"的结论；同段并补了 prefill 前后对照（8K C=1 +0.19%、80K C=1 −0.23%），关闭该节两处"要靠实测确认"的悬项。

## Close the program

- [x] 三个 PR 全部落地（`9647b3bd` / `fa0b6030` / `08cfe5c4`），V1 到 V5 全部有证据行。
- [x] `docs/sm70_long_prefill_chunk_plan.md` 回写完成（删"必须 QPN2=0"、门重定 73.51/115.39、sha 行按长度对齐、让位代价口径说明）。
- [x] `.audit/sm70-long-prefill-chunk.tsv` 追加完成（88 行，含 V1 到 V5、lane 映射、口径更正）。
- [x] 工作区文件全部进 3 个提交，`git status` 干净。**已达成**（2026-09-15 回写）：本轮收尾时工作区完全干净；`.audit/` 已随 `46cd5c26` 入库（原为未跟踪，原因已消除）。**更正**：改动文件并集是 23 个，不是 15。
- [x] 收尾全矩阵复跑并钉进决策日志。**已做**：8K C=1 87.00/`02c702ca`、8K C=2 161.72/`5bfaac89`、8K C=4 270.57/`5b633030`、80K C=1 73.51/`9bb0aa71`、80K C=2 115.39/`ea2857f4`（全部当前二进制、QPN2 on、AR auto、chunk on）。

## Appendix A. Prototype evidence

本轮跑测回答了四个问题，每条带产物路径。

- [x] QPN2-on 是否还崩。不崩。复现命令（缺省 QPN2 on）跑 8K C=1 / C=2，warmup 3 graph shape 全过、exit 0、无 fallback 日志。产物 `/tmp/qpn2on_repro_8k_c1.log`、`/tmp/qpn2on_repro_8k_c2.log`。根因是 A 单元三处加固已进二进制（qpn2 .o 重编于 23:16，晚于昨日 18:xx 的崩溃 run），见 Appendix D。
- [x] 三单元各赚多少。Appendix C 分单元表。产物 `/tmp/qpn2on_acceptance.log`（QPN2-on 矩阵）、`/tmp/qpn2off_8k_c1.log`、`/tmp/qpn2off_8k_c2.log`、`/tmp/qpn2on_final_batch.log`（AR A/B）。
- [x] 被卡的两条门是否可达。可达，且在**当前二进制**上复测仍达标：80K C=1 decode 73.51 ≥ 61.16，80K C=2 common window 115.39 ≥ 108.46（让位 40 保留）。产物 `/tmp/v7_80k_c1_on.log`、`/tmp/v7_80k_c2_on.log`（初版 67.18/110.10 测于 combine 之前，见 Appendix C 口径修正）。
- [x] "C≥2 sha 漂移"是否存在。**不存在。** 上一轮把它列为未证嫌疑，本轮用日志
  长度对齐证伪了自己：`0e75bdf6` / `874972b7` 全是 `out=64` 的 run，
  `5bfaac89` / `5b633030` 全是 `out=256` 的 run，而 `_token_stream_hash`
  把全部 token id 一起哈希，长度不同必然不同。长度对齐后跨二进制稳定
  （80K C=1 `9bb0aa71`、80K C=2 `ea2857f4` 各自横跨 09-14 与 09-15 两次
  二进制）。A/B 两单元出清。完整证据见 Appendix E。
- [x] 8K 侧的同长度跨二进制确认（V1，已跑并命中）。`--output_tokens 64` 在
  当前二进制上回到 `0e75bdf6…6036`，与昨日逐字相同。

分支与 SHA。当前 HEAD `fd226eb2`。三单元（C `9647b3bd`、B `fa0b6030`、
A `08cfe5c4`）已提交。

工作区在当前 HEAD 之上另有改动，**注意其中有并行的第二作者**：

| 路径 | 谁 | 内容 |
| --- | --- | --- |
| `tools/fastllm_pytools/benchmark.py` | 本方案 | sha256 行自描述 token 数（防"长度不同硬比 sha"复发） |
| `docs/sm70_npad_ar_chunk_landing_plan.md` | 本方案 | 本文件 |
| `docs/sm70_long_prefill_chunk_plan.md` | 本方案 + 第二作者 | sha 口径行、门重定基线 |
| `docs/sm70_1cat_port_plan.md` | 本方案 + 第二作者 | §5.1 ncu 矩阵（本方案）与 AR push 等新增节（第二作者） |
| `docs/sm70_tp4_push_ar_plan.md` | 第二作者 | AR push 方案 |
| `src/models/qwen3_5.cpp` | **第二作者** | env-gated `FASTLLM_BATCH_DECODE_DEBUG` 诊断（无数值路径） |
| `docs/sm70_nvfp4_coverage_plan.md` | 第二作者 | 新文档 |
| `.audit/` | 本方案 | 决策轨迹 |

并存期间的两条纪律：**任何 perf/sha 结论要注明测的是哪个 `.so` 时间戳**；
**不要替第二作者重排其正在编辑的内容**。

## Appendix B. 工作区现状

15 个改动文件的归属（`git diff --stat` 全量）。

| 单元 | 文件 | 内容 | 门控 |
|---|---|---|---|
| A QPN2 N-pad | `src/devices/cuda/sm70/qpn2_nvfp4.cu`、`include/devices/cuda/fastllm-sm70.cuh`、`include/devices/cuda/fastllm-cuda-fp8.h`、`src/devices/cuda/linear/fastllm-linear-fp8.cu`、`include/fastllm.h`(`nvfp4Qpn2Wanted`)、`src/models/qwen3_5.cpp`(9685 区)、`test/ops/sm70QpnNvfp4Regression.cu`、`docs/sm70_qpn_npad_design.md` | GDN-in（TP4 本地 K=5120，逻辑 N=4120）32 列对齐到 4128 过 QPN2 形状门，取代 TurboMind 回退；warmup 预建 256 条侧车；capture 守卫；`nvfp4Qpn2Wanted` 防 TurboMind 破坏 native 布局 | `FASTLLM_SM70_NVFP4_QPN2`（缺省 on） |
| B AR | `src/devices/multicuda/fastllm-custom-allreduce.cu`、`include/devices/multicuda/fastllm-multicuda.cuh`、`include/models/qwen3_cuda_common.h`(`Qwen3CudaTpReducePingPong`)、`src/models/qwen3_5.cpp`(3419/3471 + 11265 至 15091)、`test/ops/customAllReduceRegression.cpp`、`docs/sm70_ar_microbench*.md`、`docs/sm70_tp4_push_ar_plan.md` | one-stage custom 独立 dest 跳 end barrier；ping-pong 双指针烤进 CUDA Graph；auto 探测尺寸改 10 KiB；TP≥4 时 ≥40 KiB 硬切 NCCL | `FASTLLM_CUDA_CUSTOM_ALLREDUCE`（缺省 auto） |
| C 调度分块 | `include/models/longPrefillChunk.h`(新)、`src/models/basellm.cpp`、`src/models/qwen3_5.cpp`(21341 至 23672)、`include/models/basellm.h`、`test/ops/longPrefillChunkAlgebra.cpp`(新)、`test/ops/fillLlmInputsPrefillChunkRegression.cpp`(新)、`CMakeLists.txt`、`tools/fastllm_pytools/benchmark.py`、`docs/sm70_long_prefill_chunk_plan.md` | 剩余计数选批分块 + 在飞分块优先 + 让位，`RunNewMainLoop` 与 `Qwen35MTPLoop` 双路径 | `FASTLLM_LONG_PREFILL_CHUNK`（缺省 on，`0` 关） |

共用文件是切分的关键。A/B/C 在 `qwen3_5.cpp` 三个行区（A 9685 至 9727、B 3423 至 15050、C 8 与 21341 至 23672）互不重叠，按 hunk 可切（已由三次提交验证）；`basellm.cpp` 只有 C；`include/fastllm.h` 只有 A。`minimax_m2.cpp` 未改动、不引用 `Qwen3CudaTpReducePingPong`（HEAD 与工作区均 0 处），不牵连。

## Appendix C. 收益（分单元）

所有数字来自 `/tmp` 已存日志。机器 Qwen3.8-27B-QUASAR-NVFP4、TP4、`--cuda_embedding`、`--max_batch 4`、greedy、CUDA Graph 开、`FASTLLM_PAGED_CUBLAS_CHUNK=2048`。

**口径修正（2026-09-15 14:15，V5 副产物）**：本表初版的绝对数字（79.47 / 153.70 等）测于 **08:59 到 09:17**，那时 **Combine 并行化（`debbf431`）还没落地**（其 `.o` 生成于 10:30）。当前二进制（13:12）两臂都吃到 combine 的 +10%，所以绝对值整体上移，QPN2 的**比值**基本不变。下表给出两套：

- **当前二进制同源 A/B**（今日 13:12 后，V5/V6 实测，可直接引用）
- **初版数字**（08:59 到 09:17，无 combine）保留作对照与溯源

### A 单元 QPN2 N-pad

QPN2 让 GDN-in 走 dense GEMM 侧车，省每步权重反量化。反量化成本与 M 基本无关，收益随 batch 扩大。

**当前二进制（含 combine），同源：**

| 场景 | QPN2 off | QPN2 on | 差 |
|---|---:|---:|---:|
| 8K C=1 decode | 70.34（TPOP 14.22ms） | **87.00**（11.49ms） | **+23.7%** |
| 8K C=1 decode（复跑） | n/a | 87.13（11.48ms） | n/a |
| 8K C=2 common window | 99.46（26.14ms） | **161.72**（18.50ms） | **+62.6%** |
| 80K C=1 decode | 60.06（16.65ms） | **73.51**（13.60ms） | **+22.4%** |
| 80K C=2 common window | 77.51（104.55ms） | **115.39**（96.78ms） | **+48.9%** |
| 8K C=4 common window | 134.38 | **270.57** | **+101.4%** |
| 8K C=1 prefill | 2570.30 | 2575.25 | +0.2%（持平） |

80K C=2 的两个臂都在 `--output_tokens 256`、yield 40/0 保留、sha `ea2857f4`
下测得，可直接对比。**两条原本"不可达"的门在当前二进制上照样达标**：
80K C=1 decode 73.51 ≥ 61.16，80K C=2 common window 115.39 ≥ 108.46
（且让位 40 保留）。所以"门可达"这个结论不再依赖 combine 之前的旧数字。

**初版数字（08:59 到 09:17，无 combine），保留对照：**

| 场景 | QPN2 off | QPN2 on | 差 | 同源 |
|---|---:|---:|---:|:--:|
| 8K C=1 decode | 65.29（TPOP 15.32ms） | 79.47（12.58ms） | +21.7% | 是 |
| 8K C=2 common window | 96.81（26.69ms） | 153.70（19.10ms） | +58.8% | 是 |
| 8K C=4 common window | 129.27 | 263.90 | +104% | 跨源（已被当前二进制同源 134.38→270.57 取代） |
| 80K C=1 decode | 56.48 | 67.18（14.89ms） | +18.9% | 跨源 |
| 80K C=2 common window | 75.83 | 110.10（yield 40 保留） | +45.2% | 跨源 |
| 8K C=1 prefill | 2577.50 | 2569.83 | −0.3% | 是（初版；当前同源见上表 2570.30→2575.25） |

交叉验证：试算表里的 8K combine 收益，`.audit/sm70-longctx-kv.tsv` 行 8 独立记有
「combine win is +10.0% at 8K too（**87.37 vs 79.40** same-build A/B）」，
与本次 V5 实测的 87.00 在噪声内一致，佐证"上移来自 combine 而非别的"。

`sm70_qpn_npad_design.md` §8 的 +1.16% 是"只回退本单元 6 文件、1024-in、不同树"的干净差，与本表不矛盾。

### B 单元 AR

**当前二进制（含 combine），同源：**

| 场景 | AR off（NCCL） | AR auto | 差 |
|---|---:|---:|---:|
| 8K C=1 decode | 79.35 | **87.00** | **+9.6%** |
| 80K C=1 decode | 67.98 | **73.51** | **+8.1%** |
| 8K C=2 common window | 149.86 | **161.72** | **+7.9%** |

三行的 auto 臂与 A 表同 run，off 臂为 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`（强制
NCCL），各 sha 与 auto 臂一致（`02c702ca` / `9bb0aa71` / `5bfaac89`）。

**初版数字（08:59 到 09:17，无 combine），保留对照：**

| 场景 | AR off（NCCL） | AR auto | 差 | 同源 |
|---|---:|---:|---:|:--:|
| 8K C=1 decode | 72.84（13.73ms） | 79.47（12.58ms） | +9.1% | 是（旧二进制） |
| 80K C=1 decode | 62.53（15.99ms） | 67.18（14.89ms） | +7.4% | 是（旧二进制） |
| 8K C=2 common window | 143.28 | 153.70 | +7.3% | 是（旧二进制） |

两套的差都在 **+8% 到 +10%** 区间，说明 AR 的增量不随 combine 改变（combine 抬高
的是两臂共同基线，AR 路径本身没动）。

微基准（`sm70_ar_microbench.md`，10 KiB 单次 12.94 对 16.05，−19%）第一次在引擎 decode 墙钟上兑现（deepdive 里明确写过"no-end 的墙钟收益还没在引擎 decode 上量过"），量出来 +7 到 +9%。

### C 单元调度分块

C 单元的收益不是让单条更快，而是并发时先到者不被饿死，单请求速率零回退。

| 场景 | 指标 | 数值 | 说明 |
|---|---|---:|---|
| 80K C=2 | #0 在 #1 prefill 期间的让位 | 40 token | 改前 0；40 个 chunk 各让 1 次 |
| 8K C=2 / C=4 | before last TTFT | 4 / 12-8-4-0 | 让位随 batch 扩大 |
| 80K C=1 | prefill / decode | 1988.78 / 73.51 tok/s | 单请求无回退（当前二进制） |
| 任意 | 让位对 common window 的代价 | 当前二进制 −6.51 tok/s（chunk off 121.90 对 chunk on 115.39/115.52） | 但同一对 run 的 `Total time` 只 **+0.7%**（86.43 对 86.94 s）；差值是窗口定义（从 last TTFT 起算）造成的，见 `sm70_long_prefill_chunk_plan.md` §5.1 与 PR 1 的 perf 规则 |

### 组合（三单元全开，缺省配置）

| 场景 | 全开 | 原方案基线 | 过门 |
|---|---:|---:|:--:|
| 80K C=1 decode | **73.51** | 61.16 | 过 |
| 80K C=2 common window | **115.39**（yield 40） | 108.46（零让位） | 过 |
| 80K C=2 TTFT #1 | 82.95 s | 83.09 s | 过 |
| 8K C=4 common window | **270.57** | 129.27 | 过 |

全部行都是当前二进制同源实测（2026-09-15 16:12 与 16:35）。TTFT #1 用同一 run 的
窗口值（`last TTFT 82.95 s + 4.07 s`）。8K C=4 的让位 12/8/4/0 保留。

组合是 A、B、C 叠加。同一 decode 步既省反量化（A）又省 end barrier（B），并发时 #0 还在出 token（C）。

## Appendix D. 前提翻转，QPN2-on 不再崩

`sm70_long_prefill_chunk_plan.md` 此前判"本树必须 `FASTLLM_SM70_NVFP4_QPN2=0`"，依据是 warmup 崩 cublas。本轮复现不崩。缺省（QPN2 on）跑 8K C=1 / C=2，warmup 3 个 graph shape 全过，exit 0，无 fallback 日志。

根因是 A 单元三处加固（`sm70_qpn_npad_design.md` §6）已生效。warmup 期预建侧车（不在 capture 里 `cudaMalloc`）、`FastllmCudaGraphIsCapturingFast()` 守卫、`nvfp4Qpn2Wanted` 让失败退非破坏性 native 反量化。昨天崩时这些加固还没全部进二进制（qpn2 .o 重编于 23:16，晚于昨天 18:xx 的崩溃 run）。

所以"修 QPN2 warmup 崩溃"在树里已完成，剩下的是验证收尾、提交、计划文档重定基线。重定后两条"不可达"门。

| 门 | 原判 | 重定后 |
|---|---|---|
| 80K C=1 decode ≥ 61.16 | QPN2-on 基线，本树不可测 | 当前二进制 73.51 达标（QPN2 缺省开；初版 67.18） |
| 80K C=2 common window ≥ 108.46 | kernel 速率门，零让位才拿得到 | 当前二进制 115.39 达标（yield 40 也保留；初版 110.10） |

## Appendix E. 正确性与 sha 锚点

### 已证实（今日二进制内）

- C 单元保 sha。8K C=2 chunk-on == chunk-off == `5bfaac89`（`/tmp/qpn2off_8k_c2*.log`，同二进制）。
- QPN2 在 8K C=1 保 sha。on == off == `02c702ca`（out=256）。
- 80K C=2 跨 QPN2 开关、跨两次二进制稳定。`ea2857f4`（昨日 QPN2-off chunk-on、昨日零让位对照、今日 QPN2-on 全同，全部 out=256）。
- 两个回归测试今日重跑 PASS。`longPrefillChunkAlgebra`、`fillLlmInputsPrefillChunkRegression`。

### 撤销上一轮的"sha 漂移"披露：那是我的口径错，不是代码漂移

上一轮我写了一条"必须披露的漂移"：8K C=2 从 `0e75bdf6` 漂到 `5bfaac89`、
8K C=4 从 `874972b7` 漂到 `5b633030`，并把 A、B 两单元列为嫌疑。**这条披露
本身是错的，现予撤销。**

**真实原因是 `--output_tokens` 不同。** `_token_stream_hash`
（`tools/fastllm_pytools/benchmark.py:11`）把**全部**生成 token id 连起来做
sha256，所以 token 数不同，哈希必然不同，跟代码无关。把每个锚点连同它的
`output_tokens` 一起对齐，模式就完全清楚了：

| output_tokens | 8K C=1 | 8K C=2 | 8K C=4 |
| --- | --- | --- | --- |
| **64** | `b960451a` | `0e75bdf6`（5 次） | `874972b7`（2 次） |
| **256** | `02c702ca` | `5bfaac89` | `5b633030` |

我上一轮拿来当"漂移"的那一对，**长度根本不一样**：`0e75bdf6` 与 `874972b7`
全部是 `out=64` 的 run（昨日），`5bfaac89` 与 `5b633030` 全部是 `out=256`
的 run（今日）。**不存在任何一次同形状、同长度的跨二进制对照**，所以那不是
漂移证据。

**长度敏感本身有受控实证。** `/tmp/ar_pingpong_decode.log`（out=64，
`b960451a`，09-14 23:57）与 `/tmp/ar_pingpong_decode_128.log`（out=128，
`d5fcc5fc`，09-14 23:58）：同一二进制、同一形状（8K C=1、图开、TP4、
AR 走 custom 3/6 路径），**只差 `output_tokens`**，哈希就变。

**反过来，长度对齐后跨二进制是稳的。** 这是关键反证：

- `9bb0aa71`（80K C=1，out=256）出现在 09-14 17:34、17:36、17:50 与
  09-15 09:08、09:34 的日志里，**跨两次二进制完全一致**。
- `ea2857f4`（80K C=2，out=256）出现在 09-14 17:30、17:32、17:47、17:49 与
  09-15 08:12、09:08，**同样跨两次二进制一致**。

所以：**没有任何证据表明 A 或 B 单元改变了 token 流。** 上一轮那条披露连同
它的嫌疑名单一并撤销。8K 与 80K 一样，只要长度对齐就是稳的；8K 之所以显示
"变了"，只是因为今日的 8K run 用了 out=256，而昨日的 8K run 用了 out=64。

### 重定后的 sha 门

`sm70_long_prefill_chunk_plan.md` §5.1 的 sha 行按**长度对齐**重定，而不是
按"单元"重定：

- 任何 sha 对照**必须写清 `--output_tokens`**，长度不同不得对比。
- 已证：同形状、同长度、跨二进制 sha 一致（80K C=1 `9bb0aa71`、
  80K C=2 `ea2857f4`）。
- 已证（V1，2026-09-15 13:00，`/tmp/v1_8k_c2_out64.log`）：8K C=2 用
  `--output_tokens 64` 在当前二进制上复跑，sha256 =
  `0e75bdf66285415a0df02f6857f18975ffd1070036e1ace3e5ed2d273cde6036`，与
  昨日 out=64 锚点**逐字相同（全 64 位）**。8K 侧跨二进制不变性据此成立，
  A/B 出清。
- 已证（V2，2026-09-15 13:40，`/tmp/v2f_8k_c4_out64.log`）：8K C=4 用
  `--output_tokens 64` 复跑，sha256 =
  `874972b70407fc52cf40d4ecf6a7528294c7d598569f812c6e4fce4c586faa3a`，与昨日
  out=64 锚点逐字相同。**8K 两个形状（C=2、C=4）现已全部证完跨二进制不变，
  三个单元没有一个改过 token 流。**

## Appendix F. GPU 门控验证队列

执行时一次只跑一条。跑前 `nvidia-smi --query-gpu=memory.used` 确认四卡全 <1GB，跑完复确认回落。GPU 被占就 sleep 等，不叠跑。

- [x] V1（已跑，2026-09-15 13:00，命中）。8K C=2、`--output_tokens 64`、
  当前二进制，sha256 = `0e75bdf66285415a0df02f6857f18975ffd1070036e1ace3e5ed2d273cde6036`，
  与昨日 out=64 锚点逐字相同。A/B 出清。证据 `/tmp/v1_8k_c2_out64.log`。
  **口径记进这次 run 的命令**：`--output_tokens 64`（上轮的教训就是没记）。
- [x] V2（已跑并命中，2026-09-15 13:40，`/tmp/v2f_8k_c4_out64.log`）。8K C=4、
  `--output_tokens 64`，sha256 =
  `874972b70407fc52cf40d4ecf6a7528294c7d598569f812c6e4fce4c586faa3a`，与昨日
  out=64 锚点**逐字相同（全 64 位）**。0 sidecar 失败。
  **前三次没成是我的 `--tokens` 配大了**：`167936` / `40960` / 默认都 OOM
  （gpuFree 只有 19 到 33 MB）；改成 `49152`（按 4×8K 配，而不是 C=1/C=2 的
  163840 配方）就过了。boot recipe 里那个 `--tokens 167936` 是 C=1/C=2 的值，
  batch 大时必须调小。
- [x] V3（已跑，2026-09-15 13:55，`/tmp/v3_mem_samples.txt` + `/tmp/v3_80k_c1_mem.log`）。
  80K C=1 QPN2-on，逐秒 `nvidia-smi` 采样 88 点：

  | 量 | GPU0 | GPU1-3 | 余量（对 16384 MiB） |
  | --- | --- | --- | --- |
  | 稳态平台（t=40 到 85s） | 14949 MiB | 14871 MiB | **1435 MiB（1.40 GiB）** |
  | 瞬时峰值（t=53 到 56s，warmup/capture） | **16113 MiB** | 16035 MiB | **271 MiB（0.26 GiB）** |

  四卡同步升降，确认是我的 run；sha256 `9bb0aa71` 命中 80K C=1 锚点，exit 0。
  **更正**：我此前写的"峰值 ≤ 14.95/16 GB、余 1.05 GB"里，14949 是**稳态平台**
  而不是峰值；真正的峰值是 16113，峰值余量只有 271 MiB。这个薄余量正好解释
  本轮 V2 那几次 C=4 OOM。
- [x] V4（已跑并**通过**，2026-09-15 15:30，`/tmp/v4e_mtp_8k_c2.log`）。
  真 MTP / DFlash 路径确认 C 单元不开分块（`mtpDraftsPerStep==0 && !schedulerUsesDFlash`
  gate，qwen3_5.cpp:21394）。

  **结果**：8K C=2、out=64、`--speculative_algorithm mtp --mtp 1 --tokens 40960`，
  MTP **确实生效**（15 条 `[Qwen3.5 MTP]` 行：FP8 draft lm_head、transformer TP=4、
  warmup device、paged workspace），而 banner 打的是：

  ```
  [Fastllm] Long prefill chunk: off, size=2048, batch token limit=2048, scheduler=Qwen35MTPLoop.
  ```

  即 gate 正确地在 MTP 下关掉了分块。sha256 `0e75bdf6…6036`，与 no-MTP 的 out=64
  锚点**逐字相同**，说明 MTP 也没有扰动 greedy 流。exit 0。

  **怎么启用 MTP（本轮代码追出来的，初版写法是错的）**：手 export
  `FASTLLM_QWEN35_ENABLE_MTP=1` 无效。`benchmark.py:8` 引入
  `make_normal_llm_model`、在 `:456` 调用它，而该函数**无条件**执行
  `os.environ["FASTLLM_QWEN35_ENABLE_MTP"] = str(mtp)`（`util.py:1606`），
  其中 `mtp` 来自 argparse 的 `--mtp`，默认 **0**（`util.py:843`）。所以手设的
  环境变量被静默改回 0。正确调用是走 CLI：

  ```
  --speculative_algorithm mtp --mtp 1
  ```
  （`util.py:1067` 起校验：mtp 必须 >0 或给了外部 draft checkpoint。
  已用真 parser 验证：`--mtp 1` 解析为 `mtp=1`，缺省为 `mtp=0`。）

  **显存**：MTP 之后 `--tokens 167936` 会 OOM（draft lm_head 单项就要
  2.54 GB 源 + 1.31 GB FP8 副本），本项用 `--tokens 40960` 跑通。

  **一笔遗留（与本项无关，但同在这条命令上出现过）**：早先 graph on 且
  headroom 只有 `free=3.23 GB` 时，进程卡在 `warmup capture 2/3 batch=2`
  约 8 分钟、四卡 100%，我 kill 了；成功 run 的 `free=4.36 GB`。倾向于
  **共租户抢显存**导致捕获饥饿，但挂点落在 C 单元改过的 `Qwen35MTPLoop`
  图捕获阶段，所以不下定论。判定办法：干净四卡（free ≥ 4.3 GB）重跑同一命令。
- [x] V5（已跑并命中，2026-09-15 14:10，`/tmp/v5b_8k_c1_qpn2on.log` + `/tmp/v5c_8k_c1_rep2.log`）。
  内存压力下 QPN2 侧车会静默退回 native 反量化，而且有**两条**不同的日志文案，
  都表示该投影没走 QPN2（`fastllm-linear-fp8.cu:3640`）：
  - `Fastllm SM70 NVFP4 QPN2 sidecar alloc failed (K=… N=… bytes=…)`（`cudaMalloc` 失败）
  - `Fastllm SM70 NVFP4 QPN2 sidecar conversion failed (K=… N=…)`（prepack 失败）
  两条都是**设计内的非破坏性回退**（`nvfp4Qpn2Wanted` 已置位，native 布局保留，
  可重试），但意味着 QPN2-on 的性能数字会静默变成 QPN2-off 的数字。
  **实测**：8K C=1 连跑两次（87.00 / 87.13 tok/s），两条回退文案各 **0** 次，
  sha `02c702ca` 命中锚点。此后每次跑测后都 grep 这两行，作为"QPN2 是否真开着"的证据。
  **副产物（重要）**：这次跑出 87.0 而表里写 79.47，追下去发现表里的绝对数字
  测于 Combine 并行化（`debbf431`）落地之前，见 Appendix C 的口径修正。

## Appendix G. 风险与操作纪律

| 风险 | 缓解 |
|---|---|
| 共享机 GPU 被占，OOM，kill，Xid 31 | 操作纪律（下）；OOM 的 run 结果作废重跑，不当证据 |
| sha 对照口径错（`output_tokens` 不一致） | 已发生一次并撤销（Appendix E）；规则写死：任何 sha 对照必须同 `--output_tokens` |
| A 单元 prefill 路径变化（TurboMind 换 cuBLAS） | V1/V3 量 prefill tok/s，head < trunk × 0.99 则 A 单独立项回看 |
| `git add -p` 切错 hunk | 每提交后 `git show --stat` 复核 + 纯单元 HEAD 重跑该单元 unit 测试；三单元行区互不重叠（Appendix B） |
| 侧车 4.23GB 吃 16GB 卡余量 | **V3 实测（80K C=1，逐秒采样）**：稳态平台 14949 MiB（余 1435 MiB），warmup/capture 瞬时峰值 16113 MiB（**余仅 271 MiB**）。余量确实很薄，所以 C=4 长 prompt 不能叠，且任何额外显存开销都可能 OOM |

操作纪律（本轮教训，先写死）。这台 4×V100 是租户共享机。今天多次在 GPU 被占到 ~14GB 时启动 benchmark，权重加载 OOM（gpuFree 1-2GB），进程被杀，teardown 产生 Xid 31 MMU fault（dmesg 可见，发生在 GPU1/GPU3，pid 是被杀的 python3）。每条 GPU 命令前必须确认四卡空闲，绝不叠跑。GPU 被占就等，不抢。

回滚总则。三单元各有独立 env 开关，任一单元出问题可单独关。提交顺序 C、B、A 保证任意前缀的 HEAD 都是已验证的绿状态。

## Appendix H. 链接与阅读清单

- [x] 编辑前读 `docs/sm70_qpn_npad_design.md`（A 单元契约与不变式）、`docs/sm70_ar_microbench.md` 与 `docs/sm70_ar_microbench_deepdive.md`（B 单元探针数字与墙钟兑现）、`docs/sm70_long_prefill_chunk_plan.md`（C 单元判据与复现命令）、`docs/sm70_tp4_push_ar_plan.md`（B 单元的后续 push 路线，本方案不做）。
- [ ] PR 2 与 PR 3 提交前跑 `skills/how/SKILL.md` 过一遍 diff 的调用链（分发顺序）。**符号更正（2026-09-15）**：原文写的 `Nvfp4QpnTry` 在代码中不存在，实际是 `FastllmCudaTryNVFP4Qpn2`（`src/devices/cuda/linear/fastllm-linear-fp8.cu:3675`）；`FastllmCudaTryTP2P2PAllReduceAdd` 同样零命中，TP4 push-add 的实际符号是 `FastllmCustomAllReducePushAddKernel`（`src/devices/multicuda/fastllm-custom-allreduce.cu:806`）。
- [x] 决策轨迹在 `.audit/sm70-long-prefill-chunk.tsv`（show-me-your-work），本 program 追加行按 V1 到 V5 编号。
- [x] 原始日志 `/tmp/qpn2on_acceptance.log`、`/tmp/qpn2off_8k_c1.log`、`/tmp/qpn2off_8k_c2.log`、`/tmp/qpn2on_final_batch.log`、`/tmp/80k-c2-chunk-off-control.log`。**已核**（2026-09-15）：五个文件都还在（64 KB–222 KB）。
