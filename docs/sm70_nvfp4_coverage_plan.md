# 原生 NVFP4 覆盖面：收尾与几何方案

日期：2026-09-15
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP / CUDA Graph decode
上游：`docs/sm70_1cat_port_plan.md` §3 第二优先、`docs/sm70_qpn4_qpn8_review.md`、
`docs/sm70_qpn_npad_design.md`、`docs/sm70_npad_ar_chunk_landing_plan.md`
证据：`/home/nsys/dec80k_post.sqlite`（80K decode，Combine 修复后）、
`/home/nsys/dec8k.sqlite`、`/home/nsys/dec180k.sqlite`（prefill 主导）

## 0. 结论先行

1. **覆盖面主体已经落地。** 最新 trace（80K post-combine，device 0，112 token
   窗口）里 QPN2 命中 **255.98 条/token ≈ 256 全命中**，窗口里**没有**任何
   TurboMind `gemm_kernel`、没有 crop kernel。`sm70_qpn4_qpn8_review.md` §3
   的「208 命中 + 48 条 GDN-in 走 TurboMind」是旧树状态，已被 npad 侧车落地
   取代：GDN-in 现在跑 `nvfp4_qpn2_sm70_kernel` grid=129（48 条/token，
   25.64 µs/次），crop 已折叠进 QPN2 的 store 上界。decode 侧 NVFP4 投影
   **256/256 全原生**。
2. **真正的剩余缺口只有一条：lm_head。** 它是 decode 里唯一没走 NVFP4 的
   GEMM（FP16 GEMV，717.8 µs/token，与上下文无关，占 busy 5.3%）。但它是
   **权重契约变更**（checkpoint 里 lm_head 本就不在 NVFP4 集合），logits 会
   变，过不了 greedy sha256 门，只能走质量门单独立项。1Cat 的 online-QPN8
   因改变 token 轨迹在验收里被显式关闭，是直接先例。
3. **本方案的主体因此转为「既有覆盖的 per-shape 几何」。** 四个形状组实测
   421–671 GB/s，全组对 85% roof（765 GB/s）的上限是 **1.79 ms/token
   （80K ≈ 12.5%）**。第一杠杆是 N=5120 混合组、GDN-in、QKV 三组的
   occupancy（grid 160/129/112，wave 尾 2.0/1.61/1.4）。
4. **执行顺序**：U1 per-shape 探针定目标 → U2 QPN2 形状特化接线 → U3
   长上下文/并发验证。U1 任一形状探针打不到 ≥600 GB/s 就关闭本方案，
   维持「不做」的原判；达到了再谈接线。

## 1. 现状普查（2026-09-15，dec80k_post.sqlite，device 0，112 token）

窗口 1602.812 ms，14.311 ms/token（69.88 tok/s），103137 kernels，
busy 94.3%。桶级：

| 桶 | 条/token | ms/token | % busy |
|---|---:|---:|---:|
| QPN2 | 255.98 | 6.271 | 46.5% |
| attention（Split+Combine+GDN 相关） | 48.00 | 2.846 | 21.1% |
| custom AR | 128.00 | 1.925 | 14.3% |
| norm | 176.97 | 0.776 | 5.8% |
| lm_head（FP16 GEMV） | 0.99 | 0.712 | 5.3% |
| other（GDN delta/conv/AddTo 等） | 310.92 | 0.960 | 7.1% |

QPN2 按形状拆（grid = N/32，与 64 gate/up + 128 N=5120 组 + 48 GDN-in +
16 QKV 的推演逐条对上）：

| 形状组 | grid | 条/token | µs/call | 权重字节/次 | GB/s | % of 900 |
|---|---:|---:|---:|---:|---:|---:|
| MLP gate/up（K5120×N8704） | 272 | 64 | 37.34 | 25.06 MB | **671** | 74.6% |
| N=5120 组（down K4352×64 + GDN-out K1536×48 + attn-O K1536×16） | 160 | 128 | 17.64 | 加权 8.47 MB | **480** | 53.3% |
| GDN-in（K5120×N4128，npad 侧车） | 129 | 48 | 25.64 | 11.88 MB | **463** | 51.5% |
| attn QKV（K5120×N3584） | 112 | 16 | 24.49 | 10.32 MB | **421** | 46.8% |
| 合计 | — | 256 | 6.27 ms/tok | 3.424 GB/tok | 546 | 60.7% |

8K trace 同构（12.220 ms/token，QPN2 55.1% busy，AR 16.5%，attn 7.0%，
lm_head 6.2%），说明形状组的 µs/call 基本不随上下文变，占比变化全部来自
attention。

roof 账：每 token 每 rank 的 NVFP4 权重字节 = 3.424 GB（256 条之和，0.5625
系数已含 E4M3 scale）。QPN2 实测 6.27 ms → 546 GB/s；对 85% roof（765 GB/s）
的差 = **1.79 ms/token（12.5%）**。这是全组的理论上限；按形状可达性打折后，
本方案的立项门定在 **≥+3%**，目标 **≥+6%**。

## 2. 覆盖面台账

| 权重/GEMM | decode 现状 | 定性 |
|---|---|---|
| 256 条 NVFP4 投影（gate/up、down、QKV、O、GDN-in/out） | 全部 `nvfp4_qpn2_sm70_kernel` | **已覆盖** |
| lm_head | FP16 GEMV `FastllmGemvFp16Fp16Kernel2MultiRow`，717.8 µs/tok | **唯一未量化**；权重契约变更 |
| C>1 decode（M=2–16） | QPN2 门内（twoTile 路径），landing 已验 8K C=2/C=4、80K C=2 | **已覆盖** |
| prefill（M=2048 chunk） | 反量化 `FastllmCudaNVFP4Block162HalfKernel` + FP16 HMMA（cutlass h884）+ NCCL | **刻意设计，非目标** |
| QPN8 | 编译在、未接线；需 channel-FP8 权重契约，本 checkpoint 纯 NVFP4 | 不适用 |
| MoE grouped / 其他模型栈 / MTP·DFlash | 不在本路线 | 非目标 |

prefill 非目标的算力界论证：2048-token chunk 的理论 compute 下限约 0.22 s
（13.5 GFLOP/token × 2048 ÷ 125 TFLOPS），实测 ~1.0 s（22% MFU）；权重流量
3.42 GB ÷ 0.22 s = 15 GB/s，可忽略。prefill 慢不在权重格式，native NVFP4
SIMT 只会更慢。dequant-fused HMMA 同理非目标。

## 3. 杠杆排序

**L1（主杠杆）：QPN2 per-shape 几何。** 三个低带宽组的共性是 CTA 数不足、
wave 尾差（80 SM 上 grid 112/129/160 = 1.4/1.61/2.0 waves），而 gate/up
（grid 272 = 3.4 waves）已经打到 671 GB/s，证明 kernel 本体在 CTA 充足时
能到 ~75% roof。候选（全部先过探针）：

- `ChooseSplitK` 按形状分发：K=5120 现在固定 split 8；对 N=4128/3584 用
  split 16（CTA 翻倍到 258/224）。kernel 模板已实例化 split 8/16/32
  （launcher 有分支），`__shared__ float partials[SplitK][RowTiles*256]`
  在 split 32 时 32 KB，需验 occupancy。
- N-tile 16 双 tile（grid 再翻倍，GDN-in 129→258）。
- N=5120 组按 K 拆开量：K=1536 的成员（attn-O/GDN-out，4.42 MB）均摊
  17.64 µs 只有 ~251 GB/s，疑似延迟受限；与 down（12.52 MB）分开定目标。

上限合计 ~1.6–1.8 ms/token；现实立项门 ≥+3%（~0.45 ms），目标 ≥+6%
（~0.9 ms）。

**L2（附带，同 kernel 文件）：QPN2 fused gated-SiLU epilogue。** 可折叠
`FastllmSwigluKernel`（64/tok，164 µs）与 GDN 的
`RMSNormSiluMulHalf128Exact`（48/tok，167 µs），上限 ~0.33 ms。QPN4 的
`nvfp4_qpn4_gated_sm70_kernel` 可作 donor。仍按 review 的判据：先 U1 再
决定，不因「QPN4 有」而做。

**L3（单独立项）：lm_head NVFP4。** 上限 ~0.31 ms/token（权重字节减半，
717.8 → ~404 µs，按同 98% HBM 效率）。它改变 logits，greedy sha256 门
必然失败，需换成质量门（GSM8K 120/128 + ppl 对比）。**不进本路线默认
范围**，先例：1Cat online-QPN8 因 token 轨迹变化被验收显式关闭。

## 4. 实施单元（每个独立可验证）

- **U0 普查固化（已完成）**：本文件 §1 的两张表即产出，脚本口径与
  §6 相同，可复跑。
- **U1 per-shape 探针**：扩展 `/home/arproto/bw_roof.cu`（已存在）量四个
  形状组的 NVFP4 读带宽 roof；扩展 `/home/arproto/qpn2_gdn.cu` 成变体矩阵
  （split 8/16/32 × N-tile 32/16×2 × 两 tile），同一权重、M=1。产出：每形状
  的可达 GB/s 与目标 µs/call 表。**门**：GDN-in / QKV / N=5120 组任一
  ≥600 GB/s 才进 U2；全不达标则本方案关闭并记一行。
- **U2 QPN2 形状特化接线**：`ChooseSplitK` 按形状 + N-tile 变体，env
  `FASTLLM_SM70_QPN2_GEOM`（缺省 auto=现役，回滚=还原分发）。**门**：
  探针达标；greedy sha256 连跑 3 次一致且等于现锚（8K `5bfaac89`、
  8K C=4 `5b633030`，见 landing 方案 Appendix E；npad 那轮出现过
  2×`539186eb`+1×`d5fcc5fc` 的不稳样本，U2 必须把「3 次全一致」写成硬门）；
  8K/80K 墙钟 ≥+3%；prefill tok/s 变化 ≤±0.5%（npad 曾 −1.43%，不许复发）；
  8K C=2 sha 一致。
- **U3 长上下文与并发验证**：80K C=2/C=4 复测；重抓 180K **decode-only**
  trace（现有 dec180k.sqlite 是 prefill 主导，不能当 decode 普查用），
  补 §1 的 180K 行；256K 抓一次。
- **U4 lm_head NVFP4 立项评估**：只写立项卡（数值契约、质量门、回滚），
  不进本路线的实施序列。

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| prefill 回退复发（npad 曾 −1.43%） | U2 门含 prefill ±0.5%；侧车不新增（570 MB/rank 已常驻），只改 launch 参数 |
| sha 漂移归属未决（Appendix E 的 `0e75bdf6`→`5bfaac89`） | U2 的 sha 门对**现锚**，不对旧锚；3 次全一致才算过 |
| split 32 的 shared 32 KB 掉 occupancy | U1 探针里量，不带进引擎 |
| CUDA graph 捕获安全 | 侧车已预热期建立；U2 改的是已捕获图的 launch 参数，重过 warmup 3 shape（`FastllmCudaGraphIsCapturingFast` 守卫已在） |
| C=2/4 的 twoTile 路径在 npad 形状上未单独验过 | U3 覆盖 |

## 6. 取证方式

```sh
# 普查（本文件 §1 的两张表）
python3 /tmp/census.py            # dec80k_post + dec8k，device 0，112-token 窗口
# 窗口锚：FastllmCustomAllReduceKernel<__half> 128/token；按 grid=N/32 拆形状

# U1 探针
cd /home/arproto && /usr/local/cuda-12.8/bin/nvcc -O3 -arch=sm_70 -o bw_roof bw_roof.cu
./bw_roof                          # 四形状组的读带宽 roof
./qpn2_gdn <variant>               # split/N-tile 变体矩阵（M=1，同一权重）

# U2 引擎 A/B（同 landing 方案的命令）
FASTLLM_PAGED_CUBLAS_CHUNK=2048 PYTHONPATH=/home/fastllm/build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 1 --tokens 16384 --dtype auto \
  --enable_thinking false --prefix_cache false --input_tokens 8192 \
  --output_tokens 128 --batch 1 --warmup 1 --temperature 0 --top_k 1
# 对照 FASTLLM_SM70_QPN2_GEOM=auto/新值；sha256 门 3 次
```

## 7. 未验证项

- 180K/256K 的 decode-only 普查没有。现有 dec180k.sqlite 是 prefill 主导
  （NCCL AR 167.5/窗口单元、cutlass h884 318.3、无 QPN2 主循环），只能当
  prefill 路径证据。U3 重抓。
- N=5120 组内部按 K 拆分的分成员带宽没量（down vs attn-O vs GDN-out 混在
  grid=160 里），L1 对该组的目标现在是加权值。
- split 16/32 的 `partials` shared 占用与 occupancy 未量。
- 「两份 80K trace 一致」在 device-0 口径下是 15.04 vs 18.08 µs：修复前
  trace 的 device-0 max 到 593 µs，含 warmup 污染，所以「一致」目前只对
  混池口径成立。U1 探针用干净权重重新定稳态值。
- lm_head 的质量门设计（U4）未做。
- 中心估计 +4%–6% 是否随上限上移（+11.7%/+10.1%）上调，等 U1 数据。
