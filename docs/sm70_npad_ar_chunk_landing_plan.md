# SM70 三单元落地方案：QPN2 N-pad / AR / 长 prefill 分块

日期：2026-09-15
范围：`/home/fastllm` 工作区当前 15 个未提交文件（1067+/167-），Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP。
状态：代码已写完，未提交、未拆分、未逐单元验证。本方案把它拆成 3 个可独立验证的提交单元，重定验收门，并把"需要 GPU 但今天不能跑"的项排成队列。

## How to read this

这是本地未提交 diff 的落地计划，不是远程 PR 编排。三个提交单元（C 调度、B AR、A QPN2）是三个 PR 节，执行顺序 C、B、A。Execution playbook 是 `playbooks/autopilot-stack.md`：owner 在各自 worktree 构建并验证，交付一个线性提交栈由 operator 审查落地。逐 PR 验证规则，逐字如下：

> Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

Box 规则。One box is one unit of work。每个 box names the evidence（`/tmp` 日志路径、file:line、回归测试名）。Check a box only when its evidence exists。GPU 队列（Appendix E）每条跑前必须确认四卡空闲（共享机纪律见 Appendix F），GPU 被占就等，不叠跑。

## Program checklist

### Arm the program

- [ ] standing orders 写入 operator 的 standing orders 并复述进 todolist。objective 是"把工作区 15 文件按 C、B、A 三单元落地为三个可独立验证的提交，V1 到 V4 全部有证据"。
- [ ] 执行 playbook 从 installed plugin 的 `playbooks/autopilot-stack.md` 重读，不凭记忆。
- [ ] root 审计 tick 每 30 分钟一次（background bash 自唤醒，不靠 sleep），状态写 status message。

### Spawn owners

- [ ] 三个 PR 各一个 owner，在各自 worktree 从当前 HEAD 构建。三个单元行区互不重叠（Appendix A），可并行；V1 需要 GPU 时全体等四卡空闲，不叠跑。

### PR mechanics

- [ ] `git add -p` 按单元 hunk 切，每提交后在纯单元 HEAD 上重跑该单元的 unit 测试，`git show --stat` 复核没串 hunk。
- [ ] 每个 PR 的 head SHA 独立验证（双侧 perf 门 + live 日志 + sha 结论），patch-id 变化则重新验证。
- [ ] 提交顺序 C、B、A 追加进线性栈，operator 审查后落地。

### Verdict and merge

- [ ] 每个 PR 的 head SHA 拿到 clean verdict 才进栈；merge 由 operator 点击，owner 不合并。

### Boot recipe

- [ ] 每条 GPU 命令前 `nvidia-smi --query-gpu=memory.used` 确认四卡全 <1GB，跑完复确认显存回落。
- [ ] 跑测命令模板 `PYTHONPATH=build-sm70-tests/tools python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding --max_batch 4 --tokens 167936 --dtype auto --enable_thinking false --prefix_cache false`，加 `FASTLLM_PAGED_CUBLAS_CHUNK=2048` 与场景的 `--input_tokens / --output_tokens 256 / --batch / --warmup 0 / --temperature 0 / --top_k 1`。

### 队列

- [ ] V1（trunk-vs-head 8K C=2 A/B，锁死二进制，定 C≥2 sha 漂移归属，Appendix E）。PR 2 与 PR 3 的前置门。
- [ ] PR 1（C 单元）提交并验证。回滚开关 `FASTLLM_LONG_PREFILL_CHUNK=0`。
- [ ] PR 2（B 单元）提交并验证，sha 门并入 V1 结论。回滚开关 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`。
- [ ] PR 3（A 单元）提交并验证，显存余量钉死。回滚开关 `FASTLLM_SM70_NVFP4_QPN2=0`。
- [ ] 回写 `docs/sm70_long_prefill_chunk_plan.md`（删"必须 QPN2=0"，两条门重定基线 67.18 / 110.10，§5.3 复现命令去掉 QPN2 开关，sha 行按单元重定）。
- [ ] `.audit/sm70-long-prefill-chunk.tsv` 追加行（QPN2-on 不再崩、收益解锁、sha 漂移披露、OOM 碰撞教训）。
- [ ] V2、V3、V4（Appendix E）跑完并记录。

## PR 1, C 单元，长 prefill 选批分块与让位

零数值风险的调度改动先落。收益见 Appendix B 的 C 单元表：并发时 #0 不再饿死（80K C=2 让位 40 token），单请求速率零回退（80K C=1 prefill 1996.83 tok/s）。

### Stage

用 `git add -p` 按 hunk 选，只进 C 的 hunk：

- 新文件整加：`include/models/longPrefillChunk.h`、`test/ops/longPrefillChunkAlgebra.cpp`、`test/ops/fillLlmInputsPrefillChunkRegression.cpp`、`docs/sm70_long_prefill_chunk_plan.md`
- `src/models/basellm.cpp`：C hunk（remaining-count、`SelectPrefillChunkLen` 调用 2255、`PrefillOrderSortKey` 1757、yield 判据、`compactBlock` 4356、banner 1645、`IntermediatePrefillGuard` 46、`prefillRemaining` 1349/501、`ClassifyRequest` 1718）
- `src/models/qwen3_5.cpp`：C hunk（21329 区 `longPrefillChunk` gate、21387 `prefillRemaining`、22024 至 23724 MTPLoop 选批）
- `include/models/basellm.h`（`prefillRemaining` 字段）、`CMakeLists.txt`（2 个测试目标）、`tools/fastllm_pytools/benchmark.py`（`_decode_tokens_before_last_ttft`）

回滚开关：`FASTLLM_LONG_PREFILL_CHUNK=0`（关分块，调度回整段放行）。

### Verify, unit

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `longPrefillChunkAlgebra`、`fillLlmInputsPrefillChunkRegression` 在纯 C HEAD 上 PASS（今日已 PASS；纯 C 提交上重跑，确认没带进 A/B hunk）。Evidence 为两测试 stdout 的 PASS 行。

### Verify, live

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 80K C=2 实跑，banner 必须 `scheduler=Qwen35MTPLoop` 且 `Long prefill chunk: on, size=2048`，`request #0 before last TTFT` ≥ 1（预期 40）。Evidence 为运行日志的 banner 行与 before last TTFT 行。
- [ ] 8K C=4 实跑，before last TTFT 12/8/4/0。Evidence 同上。
- [ ] 跑前确认四卡空闲（Appendix F 纪律）。

### Verify, perf

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 指标 = `Batch decode (common window)` + `request #N before last TTFT`，双侧同跑 80K C=2：trunk（C 关，`FASTLLM_LONG_PREFILL_CHUNK=0`）与 head（C 开）。
- [ ] 规则：head 的 before-last-TTFT > 0，且 common window 相对 trunk 回退 ≤ 3.5 tok/s（让位代价实测 2.5 tok/s，78.35 对 75.83，留噪声余量）。
- [ ] prefill 门。80K C=1 head prefill ≥ 1971 × 0.85（方案 §5.2 下限；今日实测 1996.83，trunk 侧 V1 顺带出）。
- [ ] Evidence 为两侧运行日志的 Throughput 段。

### sha 门

- [ ] 同二进制下 chunk-on == chunk-off（已证，8K C=2 两配置同出 `5bfaac89`；80K C=2 `ea2857f4` 跨配置稳定）。纯 C 提交上复确认一次。Evidence 为两日志的 sha256 行。

### Review gate

None. 调度-only，不动 kernel，不动交互。

## PR 2, B 单元，AR 独立 dest 与 auto 分档

kernel 改动，但 `FASTLLM_CUDA_CUSTOM_ALLREDUCE` 是硬回滚。收益见 Appendix B 的 B 单元表：decode 墙钟 +7 到 +9%（8K C=1 72.84 对 79.47，80K C=1 62.53 对 67.18，同源同日）。这是微基准 19%（10 KiB 单次 12.94 对 16.05）第一次在引擎 decode 墙钟上兑现。

### Stage

- 整文件：`src/devices/multicuda/fastllm-custom-allreduce.cu`、`include/devices/multicuda/fastllm-multicuda.cuh`、`test/ops/customAllReduceRegression.cpp`、`docs/sm70_ar_microbench.md`、`docs/sm70_ar_microbench_deepdive.md`、`docs/sm70_tp4_push_ar_plan.md`
- `include/models/qwen3_cuda_common.h`（`Qwen3CudaTpReducePingPong` struct）
- `src/models/qwen3_5.cpp`：B hunk（3419/3471 `hiddenStatesReduceScratch`、11265 至 15091 decode 管线 `tpReducePing` plumbing）

回滚开关：`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0`（强制 NCCL，关掉 custom 路径）。

### Verify, unit

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `customAllReduceRegression` 在纯 B HEAD 上 PASS（含独立 dest 跳 end barrier 的数值对照）。Evidence 为测试 stdout。

### Verify, live

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 8K C=1、8K C=2 decode 实跑，确认走 custom one-stage（非 NCCL），无 hang、无 inf。Evidence 为运行日志与退出码 0。
- [ ] 跑前确认四卡空闲。

### Verify, perf

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 指标 = decode tok/s（C=1 用 `Batch decode after TTFT`，C=2 用 `common window`）。双侧：trunk（`FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` 强制 NCCL）vs head（缺省 auto）。
- [ ] 规则。head decode ≥ trunk（同源预期 +7 到 +9%，Appendix B）。若 head < trunk，回滚到 NCCL、记一行、不强行提交。
- [ ] Evidence 为两侧运行日志。

### sha 门

- [ ] 并入 Program checklist 的 V1（trunk-vs-head 8K C=2 A/B）。B 单元是 C≥2 sha 漂移的头号嫌疑（auto 分档改 allreduce 路由，求和序可致 1-ulp），V1 结论落定前本 PR 的 PR 描述只写"B 单元 sha 影响以 V1 对照为准"，不写"保 sha"。

### Review gate

None. 有 `FASTLLM_CUDA_CUSTOM_ALLREDUCE` 硬回滚；若 V1 显示 C≥2 sha 漂移归因到 B，升级为 review-gated，PR 里附 1-ulp 分析。

## PR 3, A 单元，QPN2 N-pad

收益见 Appendix B 的 A 单元表：decode +19 到 +104%（随 batch 扩大，同源行 +21.7% / +58.8%），prefill 持平（−0.3%）。这是三单元里收益最大的。

### Stage

- 整文件：`src/devices/cuda/sm70/qpn2_nvfp4.cu`、`include/devices/cuda/fastllm-sm70.cuh`、`include/devices/cuda/fastllm-cuda-fp8.h`、`src/devices/cuda/linear/fastllm-linear-fp8.cu`、`test/ops/sm70QpnNvfp4Regression.cu`、`docs/sm70_qpn_npad_design.md`
- `include/fastllm.h`（`nvfp4Qpn2Wanted`）
- `src/models/qwen3_5.cpp`：A hunk（9683 区 `OnAutoWarmupFinished` 侧车预建循环，42 行整块）

回滚开关：`FASTLLM_SM70_NVFP4_QPN2=0`（GDN-in 回 TurboMind / native）。

### Verify, unit

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `sm70QpnNvfp4Regression` 20 用例 PASS（含 GDN-in 4120 到 4128 的 canary 反向对照，`sm70_qpn_npad_design.md` §7）。Evidence: 测试 stdout 的 relL2 + canary 行。

### Verify, live

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 8K C=1 decode 实跑，warmup 3 个 graph shape 全过、无 fallback 日志、无 OOM（今日已证：sha `02c702ca`，decode 79.47）。Evidence: 运行日志。
- [ ] 跑前确认四卡空闲，跑完 `nvidia-smi` 复确认显存回落。
- [ ] 显存门（设计文档 §5 的悬项，今日已实测）：80K C=2 QPN2-on，峰值 14.95/16 GB，余 1.05 GB。写进设计文档 §5，并钉死上限：16 GB 卡不能叠 C=4 长 prompt（会 OOM）。

### Verify, perf

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] 指标 = decode tok/s。双侧：trunk（`FASTLLM_SM70_NVFP4_QPN2=0`）vs head（缺省 on）。
- [ ] 规则：head decode ≥ trunk（同源预期 +21.7%，8K C=1 65.29 对 79.47）；prefill head ≥ trunk × 0.99（实测 −0.3%，2569.83 对 2577.50，QPN2 不动 prefill）。
- [ ] Evidence: 两侧运行日志的 Throughput 段。

### sha 门

- [ ] 并入 V1。若 V1 显示 C≥2 漂移归因到 A（GDN-in 走 QPN2 后的 1-ulp），PR 描述附归因，并确认是 greedy 敏感而非计算错误（80K C=2 跨 QPN2 开关稳定 `ea2857f4` 是支持证据）。

### Review gate

None. 但 A 单元动了 GDN-in 的 prefill 路径（TurboMind 换 native 反量化加 cuBLAS）。若 V1 把 C≥2 sha 漂移归因到 A，升级为 review-gated，附 1-ulp 分析。

## Close the program

- [ ] 三个 PR 全部落地，Program checklist 的 V1 到 V4 全部有证据行。
- [ ] `docs/sm70_long_prefill_chunk_plan.md` 回写完成：删"本树必须 QPN2=0"，§5.1 两条门更新为 67.18 / 110.10（QPN2 缺省开），§5.3 复现命令去掉 `FASTLLM_SM70_NVFP4_QPN2=0`，sha 行按单元重定。
- [ ] `.audit/sm70-long-prefill-chunk.tsv` 追加：QPN2-on 不再崩（含根因）、收益解锁表、sha 漂移披露与归属、OOM 碰撞教训。
- [ ] 工作区 15 文件全部进 3 个提交，`git status` 干净（`.audit/`、`docs/` 其余方案文件、测试文件按本次范围一并提交或明确留在工作区并记录原因）。
- [ ] 最终一次全矩阵（QPN2 on、AR auto、chunk on）复跑 80K C=1/C=2 + 8K C=2/C=4，四个 sha 与 tok/s 钉死进决策日志，作为本 program 的收尾证据。

## Appendix A, 工作区现状

15 个改动文件的归属（`git diff --stat` 全量）。

| 单元 | 文件 | 内容 | 门控 |
|---|---|---|---|
| A QPN2 N-pad | `src/devices/cuda/sm70/qpn2_nvfp4.cu`、`include/devices/cuda/fastllm-sm70.cuh`、`include/devices/cuda/fastllm-cuda-fp8.h`、`src/devices/cuda/linear/fastllm-linear-fp8.cu`、`include/fastllm.h`(`nvfp4Qpn2Wanted`)、`src/models/qwen3_5.cpp`(9683 区)、`test/ops/sm70QpnNvfp4Regression.cu`、`docs/sm70_qpn_npad_design.md` | GDN-in（TP4 本地 K=5120，逻辑 N=4120）32 列对齐到 4128 过 QPN2 形状门，取代 TurboMind 回退；warmup 预建 256 条侧车；capture 守卫；`nvfp4Qpn2Wanted` 防 TurboMind 破坏 native 布局 | `FASTLLM_SM70_NVFP4_QPN2`（缺省 on） |
| B AR | `src/devices/multicuda/fastllm-custom-allreduce.cu`、`include/devices/multicuda/fastllm-multicuda.cuh`、`include/models/qwen3_cuda_common.h`(`Qwen3CudaTpReducePingPong`)、`src/models/qwen3_5.cpp`(3419/3471 + 11265 至 15091)、`test/ops/customAllReduceRegression.cpp`、`docs/sm70_ar_microbench*.md`、`docs/sm70_tp4_push_ar_plan.md` | one-stage custom 独立 dest 跳 end barrier；ping-pong 双指针烤进 CUDA Graph；auto 探测尺寸改 10 KiB；TP≥4 时 ≥40 KiB 硬切 NCCL | `FASTLLM_CUDA_CUSTOM_ALLREDUCE`（缺省 auto） |
| C 调度分块 | `include/models/longPrefillChunk.h`(新)、`src/models/basellm.cpp`、`src/models/qwen3_5.cpp`(21329 至 23724 + 21387)、`include/models/basellm.h`、`test/ops/longPrefillChunkAlgebra.cpp`(新)、`test/ops/fillLlmInputsPrefillChunkRegression.cpp`(新)、`CMakeLists.txt`、`tools/fastllm_pytools/benchmark.py`、`docs/sm70_long_prefill_chunk_plan.md` | 剩余计数选批分块 + 在飞分块优先 + 让位，`RunNewMainLoop` 与 `Qwen35MTPLoop` 双路径 | `FASTLLM_LONG_PREFILL_CHUNK`（缺省 on，`0` 关） |

共用文件是切分的关键：A/B/C 在 `qwen3_5.cpp` 三个行区（A 9683、B 3419/3471 + 11265 至 15091、C 21329 至 23724）互不重叠，`-p` 按 hunk 可切；`basellm.cpp` 只有 C；`include/fastllm.h` 只有 A。`minimax_m2.cpp` 未改动、不引用 `Qwen3CudaTpReducePingPong`（HEAD 与工作区均 0 处），不牵连。

## Appendix B, 收益（分单元）

所有数字来自 `/tmp` 已存日志。机器 Qwen3.8-27B-QUASAR-NVFP4、TP4、`--cuda_embedding`、`--max_batch 4`、greedy、CUDA Graph 开、`FASTLLM_PAGED_CUBLAS_CHUNK=2048`。同源 = 同一天同一二进制（今日 10:11 重链后）的 A/B；跨源 = off 侧是昨日二进制，方向可比、幅度含 run-to-run 抖动。

### A 单元 QPN2 N-pad

QPN2 让 GDN-in 走 dense GEMM 侧车，省每步权重反量化。反量化成本与 M 基本无关，收益随 batch 扩大：

| 场景 | QPN2 off | QPN2 on | 差 | 同源 |
|---|---:|---:|---:|:--:|
| 8K C=1 decode | 65.29（TPOP 15.32ms） | 79.47（12.58ms） | +21.7% | 是 |
| 8K C=2 common window | 96.81（26.69ms） | 153.70（19.10ms） | +58.8% | 是 |
| 8K C=4 common window | 129.27 | 263.90 | +104% | 跨源 |
| 80K C=1 decode | 56.48 | 67.18（14.89ms） | +18.9% | 跨源 |
| 80K C=2 common window | 75.83 | 110.10（yield 40 保留） | +45.2% | 跨源 |
| 8K C=1 prefill | 2577.50 | 2569.83 | −0.3% | 是 |

`sm70_qpn_npad_design.md` §8 的 +1.16% 是"只回退本单元 6 文件、1024-in、不同树"的干净差，与本表不矛盾。以同源行（+21.7% / +58.8%）为准。

### B 单元 AR

| 场景 | AR off（NCCL） | AR auto | 差 | 同源 |
|---|---:|---:|---:|:--:|
| 8K C=1 decode | 72.84（13.73ms） | 79.47（12.58ms） | +9.1% | 是 |
| 80K C=1 decode | 62.53（15.99ms） | 67.18（14.89ms） | +7.4% | 是 |
| 8K C=2 common window | 143.28 | 153.70 | +7.3% | 是 |

微基准（`sm70_ar_microbench.md` 10 KiB 单次 12.94 对 16.05，−19%）第一次在引擎 decode 墙钟上兑现（deepdive 里明确写过"no-end 的墙钟收益还没在引擎 decode 上量过"），量出来 +7 到 +9%。

### C 单元调度分块

C 单元的收益不是让单条更快，而是并发时先到者不被饿死，单请求速率零回退：

| 场景 | 指标 | 数值 | 说明 |
|---|---|---:|---|
| 80K C=2 | #0 在 #1 prefill 期间的让位 | 40 token | 改前 0；40 个 chunk 各让 1 次 |
| 8K C=2 / C=4 | before last TTFT | 4 / 12-8-4-0 | 让位随 batch 扩大 |
| 80K C=1 | prefill / decode | 1996.83 / 67.18 tok/s | 单请求无回退 |
| 任意 | 让位对 common window 的代价 | −2.5 tok/s（78.35 对 75.83） | 窗口定义天生奖励零让位，见 `sm70_long_prefill_chunk_plan.md` §5.1 |

### 组合（三单元全开，缺省配置）

| 场景 | 全开 | 原方案基线 | 过门 |
|---|---:|---:|:--:|
| 80K C=1 decode | 67.18 | 61.16 | 过 |
| 80K C=2 common window | 110.10（yield 40） | 108.46（零让位） | 过 |
| 80K C=2 TTFT #1 | 82.57 s | 83.09 s | 过 |
| 8K C=4 common window | 263.90 | 129.27 | 过 |

组合是 A、B、C 叠加：同一 decode 步既省反量化（A）又省 end barrier（B），并发时 #0 还在出 token（C）。

## Appendix C, 前提翻转，QPN2-on 不再崩

`sm70_long_prefill_chunk_plan.md` 此前判"本树必须 `FASTLLM_SM70_NVFP4_QPN2=0`"，依据是 warmup 崩 cublas。本轮复现不崩：缺省（QPN2 on）跑 8K C=1 / C=2，warmup 3 个 graph shape 全过，exit 0，无 fallback 日志。

根因是 A 单元三处加固（`sm70_qpn_npad_design.md` §6）已生效：warmup 期预建侧车（不在 capture 里 `cudaMalloc`）、`FastllmCudaGraphIsCapturingFast()` 守卫、`nvfp4Qpn2Wanted` 让失败退非破坏性 native 反量化。昨天崩时这些加固还没全部进二进制（qpn2 .o 重编于 23:16，晚于昨天 18:xx 的崩溃 run）。

所以"修 QPN2 warmup 崩溃"在树里已完成，剩下的是验证收尾、提交、计划文档重定基线。重定后两条"不可达"门：

| 门 | 原判 | 重定后 |
|---|---|---|
| 80K C=1 decode ≥ 61.16 | QPN2-on 基线，本树不可测 | 67.18 达标（QPN2 缺省开） |
| 80K C=2 common window ≥ 108.46 | kernel 速率门，零让位才拿得到 | 110.10 达标（yield 40 也保留） |

## Appendix D, 正确性与 sha 锚点

### 已证实（今日二进制内）

- C 单元保 sha：8K C=2 chunk-on == chunk-off == `5bfaac89`（`/tmp/qpn2off_8k_c2*.log`，同二进制）。
- QPN2 在 8K C=1 保 sha：on == off == `02c702ca`。
- 80K C=2 跨 QPN2 开关、跨两次二进制稳定：`ea2857f4`（昨日 QPN2-off chunk-on、昨日零让位对照、今日 QPN2-on 全同）。
- 两个回归测试今日重跑 PASS：`longPrefillChunkAlgebra`、`fillLlmInputsPrefillChunkRegression`。

### 必须披露的漂移（本轮复核抓到）

工作区二进制在昨晚 23:16 至 01:59 重编（qpn2 / allreduce / qwen3_5 / basellm 的 .o）加今日 10:11 重链，晚于 `0e75bdf6`（昨日 18:21）的测量。因此：

- 8K C=2 sha 从 `0e75bdf6`（旧二进制）漂到 `5bfaac89`（当前）。
- 8K C=4 sha 从 `874972b7`（旧二进制）漂到 `5b633030`（当前）。
- 此前完成声明把 `0e75bdf6` / `874972b7` 当作"sha 全不变"的证据，那两条是旧二进制锚点，已过时。

按单元排除：C 排除（§已证保 sha）；A 可疑但无法定论（`Nvfp4QpnCanRun(1,5120,4120)=0`、pad 后 =1，M=1 decode 时 GDN-in 走 QPN2，理论可 1-ulp 差；但 80K C=2 跨 QPN2 开关稳定，长 prompt 下未翻 greedy token）；B 最可疑（auto 分档改 allreduce 路由，求和序可致 1-ulp；但 8K C=2 decode 消息 10 KiB 在 40 KiB 硬切线以下仍走 custom，理论也该稳）。

结论：停 GPU 状态下无法定论，也不靠猜。V1（trunk-vs-head 8K C=2 A/B，锁死二进制）在 B/A 提交前必须拿到。落定前不把"整个工作区保 sha"写成已证，只写"C 单元保 sha（已证）；A/B 在 C≥2 的 sha 影响待 V1"。这是本轮复核对之前完成声明的实质修正。

## Appendix E, GPU 门控验证队列

执行时一次只跑一条。跑前 `nvidia-smi --query-gpu=memory.used` 确认四卡全 <1GB，跑完复确认回落。GPU 被占就 sleep 等，不叠跑。

- [ ] V1（最高优先，PR 2/PR 3 前置）：trunk-vs-head 8K C=2 A/B，锁死二进制。trunk = 5 个已提交 commit，三单元全关（`FASTLLM_SM70_NVFP4_QPN2=0` + `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` + `FASTLLM_LONG_PREFILL_CHUNK=0`）。head = 当前工作区。各跑 8K C=2，记 sha + common window。判读：若 trunk sha == 旧锚点 `0e75bdf6` 而 head == `5bfaac89`，漂移 100% 来自三单元之一，再逐个开 A/B/C 二分定位（C 已排除）。定 Appendix D 的归属。
- [ ] V2：C=4 隔离。8K C=4 QPN2-on chunk-off 与 QPN2-off chunk-on（V1 已含后者）。定 `874972b7 对 5b633030` 归属。
- [ ] V3：80K C=1 QPN2-on 显存峰值 + 余量复确认（今日已有 14.95/16 GB 数据，V3 换干净 GPU 复跑钉死）。
- [ ] V4：真 MTP / DFlash 路径确认 C 单元不开分块（`mtpDraftsPerStep==0 && !schedulerUsesDFlash` gate，qwen3_5.cpp 21329）。跑 MTP 8K C=2，确认 banner 无 `Long prefill chunk: on`、token 流与改动前一致。

## Appendix F, 风险、回滚与操作纪律

| 风险 | 缓解 |
|---|---|
| 共享机 GPU 被占，OOM，kill，Xid 31 | 操作纪律（下）；OOM 的 run 结果作废重跑，不当证据 |
| C≥2 sha 漂移归因未定（Appendix D） | V1 在 B/A 提交前拿到对照；定论前 PR 描述不写"全 diff 保 sha" |
| A 单元 prefill 路径变化（TurboMind 换 cuBLAS） | V1/V3 量 prefill tok/s，head < trunk × 0.99 则 A 单独立项回看 |
| `git add -p` 切错 hunk | 每提交后 `git show --stat` 复核 + 纯单元 HEAD 重跑该单元 unit 测试；三单元行区互不重叠（Appendix A） |
| 侧车 4.23GB 吃 16GB 卡余量 | 实测余 1.05GB（PR 3 live 门）；文档化"不能叠 C=4 长 prompt" |

操作纪律（本轮教训，先写死）：这台 4×V100 是租户共享机。今天多次在 GPU 被占到 ~14GB 时启动 benchmark，权重加载 OOM（gpuFree 1-2GB），进程被杀，teardown 产生 Xid 31 MMU fault（dmesg 可见，发生在 GPU1/GPU3，pid 是被杀的 python3）。每条 GPU 命令前必须确认四卡空闲，绝不叠跑。GPU 被占就等，不抢。

回滚总则：三单元各有独立 env 开关，任一单元出问题可单独关。提交顺序 C、B、A 保证任意前缀的 HEAD 都是已验证的绿状态。

## Appendix G, 停测后的诚实边界

为安全停掉了一切 GPU 跑测（共享机 Xid 31 之后）。已拿到的同源 A/B（Appendix B 标"是"的行）是停测前在 GPU 空闲窗口跑的，有效。未拿到：V1 到 V4，排进 Appendix E 队列，等 GPU 空闲逐条跑。方案里所有预期数字都标了同源/跨源，没有一条是编的；所有已证都带 `/tmp` 日志路径或 file:line。不把整个工作区保 sha 写成已证（Appendix D 只证到 C 单元）。
