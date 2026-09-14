# Qwen3.8-27B-NVFP4 / SM70 / TP4 / no-MTP: 1Cat-vLLM vs FastLLM 路径对照

日期：2026-09-14
对照口径：`docs/sm70_concurrency_port_plan.md` 的合同，但只取**当前这一条**路线——
Qwen3.8-27B-QUASAR-NVFP4、4×V100、TP4、no-MTP。DeepSeek/GLM/MXFP4/AWQ-QPN
等别的模型栈不进入本表。

## 0. 冻结合同

两边的模型是同一个。`/data/models/Qwen3.8-27B-QUASAR-NVFP4/config.json`：

| 项 | 值 |
|---|---|
| hidden_size | 5120 |
| intermediate_size | 17408（gated，K5120×N8704/TP） |
| num_hidden_layers | 64 |
| head_dim / GQA | 256 / 24:4 |
| full_attention_interval | 4（其余 GDN 线性层） |
| linear_* | 48 v-heads ×128，16 k-heads，conv 4 |
| experts | **无**（dense 模型） |
| 量化 | `nvfp4-pack-quantized`，group 16，E4M3 scale，ignore lm_head |

TP4 后每条投影的本地形状，以及它们能否过 FastLLM 的 QPN 形状门：

| 投影 | K | N | QPN2 门（N%32,K%64,M≤32） |
|---|---:|---:|---|
| attn QKV | 5120 | 2048 | 过 |
| attn O | 1536 | 5120 | 过 |
| MLP gate/up | 5120 | 8704 | 过 |
| MLP down | 4352 | 5120 | 过 |
| GDN in | 5120 | **2608** | N%32=16，**不过** → 走 TM（TM 内部 pad 到 2624 再 crop） |
| GDN out | 1536 | 5120 | 过 |

这就是「GDN N=4120→4128 padding」那件事在 TP4 下的实际形态：FastLLM 由
TurboMind 的 `nvfp4Scratch` + `CropNvfp4OutputKernel` 承担，不是 QPN2。

## 1. 1Cat 的录取路线与每 token 预算

来源：`1Cat-vLLM/docs/design/sm70_qwen38_nvfp4_decode.md`（冻结路由表 + 逐 token
profile）、`sm70_qwen38_qpn8_decode.md`、`sm70_concurrency_integration_audit.md`。

71.3 tok/s / 14.017 ms 那条（native sampler）的每 token GPU 服务时间：

| 组件 | ms/token | 谁做 |
|---|---:|---|
| NVFP4 gate/up | 2.762 | TurboMind N32 lookahead1 |
| **QPN8 split-16**（FP8 down） | 2.234 | QPN8 |
| NVFP4 down | 1.706 | TurboMind N32 lookahead2 |
| **TP all-reduce** | 1.697 | custom AR |
| QPN8 split-12（FP8 output） | 0.895 | QPN8 |
| **E4M3 XQA p64** | 0.587 | Flash-V100 |
| QPN8 fused gate/up | 0.508 | QPN8 |
| TurboMind FP8 dense/LM head | 0.407 | TurboMind |

80.6 tok/s / 12.403 ms 那条（C++17 sampler sidecar）把 NVFP4 gate/up 与 down
换成 **QPN4**（2.165 / 1.441 ms），AR 变成 pack32（1.587 ms）。两条都跑在
`FULL_AND_PIECEWISE` CUDA graph 上，graph 内 kernel 覆盖 93–95%，每 rank 每
token 仍发射约 1062–1141 个 kernel。

关键点：**1Cat 的 14 ms 不是靠单个大 kernel，而是靠「所有投影都在图内、都在
SM70 原生路径、通信也进图」**。它自己写明 headroom 仍在 batch-one 的发射几何
与串行投影/通信链上。

## 2. 已迁：逐项在 FastLLM 代码里确认

| 1Cat 能力 | FastLLM 落点 | 状态 |
|---|---|---|
| NVFP4 QPN2 GEMM | `src/devices/cuda/sm70/qpn2_nvfp4.cu` + `fastllm-sm70.cuh` | 内核在；**已接 Linear** |
| TurboMind W4A16 NVFP4 | `src/devices/cuda/awq_sm70/fastllm-awq-sm70.cu:GemmNvfp4` | 在，是 QPN2 之后的第一回退 |
| SM70 s884 registry | `awq_sm70/tm_registry_sm70.cu` 注册 `sm70_884_{4,8,16}` | 在（但见 §3.5 的 tactic 缺口） |
| Native paged attn | `attention/paged/fastllm-paged-attention-native.cu` | 在；FlashInfer 在 SM70 关 |
| E4M3 / FP4 KV 支持 | 同上，`FP8_E4M3` / `FP4_E2M1` 分支 | **代码在，默认没开**（见 §3.2） |
| GDN fused 小算子 | `src/models/qwen3_5.cpp`（conv+SiLU、recurrent 等） | 在 |
| Custom AllReduce | `multicuda/fastllm-custom-allreduce.cu` | 代码在，push 路径 TP4 进不去 |
| NVFP4 persistent scratch | `awq_sm70` 的 `nvfp4Scratch` + crop | 在；就是它避开了 malloc/NCCL 死锁 |
| MTP 框架 | `qwen3_5.cpp` / `qwen3_5.h` 有 DFlash 风控骨架 | 有骨架，非 1Cat DFlash2 |

QPN2 的接法值得单独记：FastLLM **没有**原地改写 `cudaData`，而是新开一个
`Data::nvfp4Qpn2Packed` sidecar（见 `include/fastllm.h:522`、`fastllm.cpp`
的三处 free）。`Nvfp4QpnPrepareFromNative` 把 interleaved `[N,(K/16)*12]` 转成
`[codes | raw E4M3]`，native 布局留给大 M prefill。分发顺序在
`FastllmCudaHalfMatMulFloatNVFP4Block16`（`fastllm-linear-fp8.cu:3911`）：
**QPN2 → TurboMind → Marlin → GEMV/cuBLAS**。

## 3. 缺口：按对当前 50.3 tok/s 的影响排序

### P0-1 SM70 CUDA Graph 被策略关掉 —— 收益最大

- `tools/fastllm_pytools/util.py:599` 的自动开关要求
  `compute capability > 7.5`；V100 是 7.0，所以日志打印
  `Qwen3.5 auto CUDA graph disabled`。
- C++ 侧 `src/fastllm.cpp:360` 只读 `FASTLLM_CUDA_GRAPH`，**没有** arch 门。
  也就是说这是 host 侧策略，不是内核不能跑。
- `qwen4_exp.cpp:7956` 的图条件还额外要求
  `hiddenStates.dims[1]==1 || mtpTargetGraph`，并用 `FASTLLM_CUDA_MOE_CACHE_MAX_BATCH`
  限行数。
- 工作树里已经有 SM70 graph 的在制品：`qwen3_5.cpp` 的 GDN 分支现在会在捕获期
  遇到 request-owned cache 时打警告并 `FastllmCudaSetThreadError()` 退出，
  `fastllm-cuda.cu` 把 cuBLAS handle/workspace 从进程级全局改成 thread_local
  （注释直指 SM70 捕获后 fault）。方向正确，但还没到「默认可开」。

1Cat 那句「graph 是 14 ms 的前提」和「每 token 1062–1141 kernel」是同一件事：
14 ms 里 launch 税占大头，没有图就必然落后。

### P0-2 Flash-V100 XQA + E4M3 KV —— 唯一整棵没迁的 attention

`grep -rl 'XQA|flash_v100|flash-attention-v100' src/ include/ test/` 在 FastLLM
**零命中**。1Cat 的 `flash-attention-v100/kernel/` 有
`flash_decode_paged.cu`、`fp8_kv_bridge.cu`、`fused_mha_forward_paged.cu`
等，对应 0.587 ms/token 的 p64 路由。

需要给 P0-2 分层，别混为一谈：

| 子项 | FastLLM 状态 |
|---|---|
| E4M3 KV 存储/解析 | **已有**（`paged-attention-native.cu` 有 `FP8_E4M3`、`FP4_E2M1` 分支，还有 SM70 专属 `FASTLLM_PAGED_SM70_GQA_D256_DECODE`） |
| 80k 跑的是 FP16 还是 E4M3 | 取决于 `kv_cache_dtype`，默认 `"auto"`；要显式设才走 E4M3 |
| XQA p64 / p128 分区 | 没有；只有 shared-GQA / per-Q-head 路径 |
| paged→contiguous FP8 bridge | 没有 |

所以 P0-2 真正缺的是 **XQA 的 p64/p128 partition + decode 专属 kernel**，
E4M3 KV 本身是接线问题。1Cat 实测 p64 相对 p128 再省 0.365 ms/token，
相对 p256 省约 1.68 ms——但前提是 attention 已经能进图。

### P0-3 QPN4 M=1 decode — 整份未迁

`nvfp4_qpn4_sm70.cu` 1111 行在 1Cat；FastLLM 无同名/近名文件。1Cat 的 QPN4
有两个 FastLLM 完全没有的东西：

- `NAcc`（2/4 条 accumulator 链）+ per-shape `lookahead`（prefill codes 预取）；
- `nvfp4_qpn4_silu_and_mul_sm70_kernel` 与 `nvfp4_qpn4_gated_sm70_kernel`——
  融合 gate/SiLU/up 的 epilogue。

对当前数字：1Cat 在 NVFP4 gate/up（2.762→2.165）和 down（1.706→1.441）上
一共省约 0.86 ms/token，约占 14 ms 的 6%。FastLLM 现在这两条走 TurboMind。

### P0-4 QPN2 fused gated SiLU — 内核在 1Cat，FastLLM 没有 gated 变体

`grep 'gated|silu|swiglu' src/devices/cuda/sm70/qpn2_nvfp4.cu` → **零命中**。
1Cat 有 `nvfp4_qpn2_gated_sm70_kernel`（一行内做 gate+SiLU+up，并沿用 SM70
`silu_and_mul` 的两次舍入契约）。

FastLLM 的 QPN2 只是纯 GEMM，MLP 仍是 `FastllmCudaSilu` → `FastllmCudaMul`
（`qwen3_5.cpp:96` 一带）。QPN2 在 80k 上只吃 M≤32，所以这项的收益主要落在
短-中上下文 decode，不是长上下文。

### P1-5 QPN8 编了但没接 Linear —— 对这颗权重不是主路径

`Fp8QpnGemm` / `Fp8QpnPrepare` 在 `fastllm-sm70.cuh` 有声明、`qpn8_fp8.cu`
有实现、`CMakeLists.txt` 有编译、`sm70QpnFp8Regression` 8/8 通过，
**但 `src/` 下除 sm70 目录外零调用**。当前 checkpoint 是纯 NVFP4，
`lm_head` 被 ignore，所以 QPN8 对这颗权重确实不是主路径。

同时缺的还有 1Cat 的 QPN8 特化，这些在移植时被刻意排除：

- `gated_pair`（FP8 融合 gate/SiLU/up）；
- `ba_split` / split-12、split-16 的形状特化；
- `hc_*`（HyperConnection）。

1Cat 的 split-12 只在 `K1536×N5120` 上赢，省约 12.7 µs/token × 64 个 output
投影，是形状级的小账。混 FP8 的 Qwen3.8 才刚需，那时再接。

### P1-6 Custom AR 仍只开 TP2，且 SM70 ForceSync 没关

两处硬门：

- `CustomArUsePushAdd`（`fastllm-custom-allreduce.cu:980`）与 `wantPush`
  （`:2332`）都写死 `devices.size() == 2`；
- `CustomArPushEnabledFromEnv()` 默认 **false**，要显式设 env。

`basellm.cpp` 的 `AutoWarmupFinishGuard` 里明确：
`if (FastllmCudaRuntimeArch() >= 75) FastllmCudaSetNcclForceSync(false);`
——SM70 被排除，注释理由是 SM70 长 prefill 会在 warmup 后首次增长
chunked-attention / dequant scratch，真 cudaMalloc 只同步当前 GPU，会跨 rank
死锁。所以 SM70 上稳态 decode 一直带着这层同步。

1Cat 的 AR 预算是 1.697 ms/token（native 路线）/ 1.587 ms（pack32）。这是
P1 里最大的一块，但被 P0-1 卡住：图先起来，才好把 ForceSync 收掉。

### P1-7 fused AllReduce + RMSNorm

`grep 'AllReduce.*RMSNorm|reduce_scatter|ReduceScatter'` → **零命中**。
1Cat 8k prefill 5500 计划里有 reduce-scatter/RMS/all-gather 融合，FastLLM 没有。
prefill 优先级低于 decode。

### P2-8 CUTLASS SM70 prefill GEMM / bounded dispatch / DFlash2

- 1Cat 的 `qwen38_prefill_cutlass.cu` 只有 130 行，是形状级特化；
  FastLLM 8k 已 4947、80k 2011 tok/s（1Cat 64k 才 1453），**长文 prefill 这边
  不落后**，不是第一刀。
- QPN2 的 M≤32 bounded dispatch，FastLLM 已经和 1Cat 同精神实现
  （`FastllmCudaTryNVFP4Qpn2` 里 `n>32` 直接返回 false，注释写着不要把 80k 行
  塞进这个 kernel）。
- DFlash2 / MTP5 / Philox top-20 sampler：1Cat 接受线是投机 decode（12.4 ms
  /round 那条）。FastLLM 有 MTP 骨架但没有等价 draft 与 exact-Philox chunked
  sampler。这是另一条产品线，不是 80k no-MTP 的阻塞项。

### P2-9 TurboMind small-N HMMA 的精确形状 tactic

这一条值得单独列，因为它是「TurboMind 已有但被削过」的缺口。两边同名文件
行数差得很明显：

| 文件 | FastLLM | 1Cat |
|---|---:|---:|
| `sm70_884_4.cu` | 95 | **171** |
| `sm70_884_8.cu` | 28 | **81** |
| `operand_sm70_s884.h` | 缺 `Operand_A_Swizzle_8x64` | 有 |

1Cat 在 `sm70_884_4.cu` 里额外注册了：

- `ExactMKernelImpl<...,5>` / `ExactMnkKernelImpl<...,1,8704,5120>` 等
  **小 N 精确形状**（AWQ M=5/8：11.3→9.0 ms，以及 NVFP4 M=1 的
  `N8704/K5120`、`N5120/K4352` 变体）；
- `Qwen38Nvfp4W2CacheBKernelImpl`（W2 `n2560/k160` 的 cache-B）；
- `Qwen38Nvfp4W13TailN64KernelImpl`（W13 N=320 的 N64 tail）。

这三个是给 Qwen3.8 NVFP4 的 **MoE** 形状写的（`num==512`）。但同一份
`ExactMnkKernelImpl<...,1,8704,5120>` / `<...,1,5120,4352>` 正是 dense
Qwen3.8 的 MLP gate/up 与 down。FastLLM 缺 `Operand_A_Swizzle_8x64`，
所以这批 tactic 现在连注册都编不过去。

## 4. 数字对账

| | FastLLM 80k | 1Cat 相近合同 |
|---|---|---|
| Prefill | **2011 tok/s** @80k | 1453 @64k；8k FP8 ~5170 |
| Decode | **50.3 tok/s / 19.9 ms** | **~65–71 tok/s / 14–15 ms**（短到 64k） |

FastLLM 自己的长 prompt 实测（port plan §16.1，C=1）钉了稳态：
8K → 71.8 tok/s(13.9 ms)、80K → 59.98 tok/s(16.67 ms)、180K → 50.13 tok/s
(19.95 ms)。也就是说 **50.3 tok/s 对应的是 180K 上下文**，而 1Cat 的 ~71 是
短-中上下文。这个对照必须说清楚，否则会把「长上下文 KV 带宽」误当成
「kernel 缺口」。

按 profile 把 50 → 71 的 5.9 ms 差拆开（用 1Cat 的分项当量级参考）：

| 差项 | 量级 |
|---|---:|
| CUDA graph（launch 税，图内约 1062 kernel） | 最大，且是别的项的前提 |
| TP all-reduce 没进图、ForceSync 未收 | ~1.6–1.7 ms |
| Flash-V100 XQA p64 相对 native paged | ~0.6 ms + 省掉的分区开销 |
| QPN4 / fused gate-SiLU | ~0.9 ms |
| E4M3 KV 少一半 KV 带宽 | 长上下文才显著 |

## 5. 建议顺序

1. **P0-1 SM70 CUDA Graph 默认开**（先把 GDN/cuBLAS 捕获安全收口——工作树已经
   在补这两处）。这是唯一同时解锁 P1-6 的前置。
2. **P1-6 push AR 放宽到 TP4 + 收 ForceSync**（图起来之后）。
3. **P0-3 QPN4 M=1 + P0-4 QPN2 gated SiLU**（同一次搬迁，共用 sidecar
   框架）。
4. **P0-2 Flash-V100 XQA p64**；同时把 `kv_cache_dtype` 的 E4M3 接线补上，
   并在 80k/180K 上验证。
5. P2-9 的 `Operand_A_Swizzle_8x64` + ExactMnk tactic（顺手，且对 AWQ 与
   NVFP4 M=1 都有用）。
6. P1-5 QPN8 接 Linear 推迟到有混 FP8 的 checkpoint 时。

## 6. 取证方式

全表结论都来自本机静态核对，可复现：

```sh
# 内核清单对齐
wc -l /home/1Cat-vLLM/csrc/sm70_turbomind/ops/*.cu \
      /home/fastllm/src/devices/cuda/sm70/*.cu

# QPN8 是否接线
grep -rn 'Fp8Qpn' /home/fastllm/src /home/fastllm/include | grep -v 'cuda/sm70/'

# graph 策略
sed -n '589,600p' /home/fastllm/tools/fastllm_pytools/util.py
sed -n '3962,3976p' /home/fastllm/src/models/basellm.cpp

# push AR 门
sed -n '980,981p;2332,2333p' /home/fastllm/src/devices/multicuda/fastllm-custom-allreduce.cu

# small-N tactic 缺口
diff /home/fastllm/third_party/turbomind/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu \
     /home/1Cat-vLLM/csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu
```

未验证项（本轮没有 GPU 侧运行时证据，不要当结论用）：QPN2 在 80k 的实际占比、
80k 那次跑的是 FP16 还是 E4M3 KV、开 graph 后 SM70 的实际死锁面。
