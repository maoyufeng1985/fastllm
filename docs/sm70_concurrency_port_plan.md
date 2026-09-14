# FastLLM SM70 并发>1 优化搬运方案

日期：2026-09-12  
状态：部分实施（见第 16 节「实施记录」）  
范围：把 1Cat-vLLM 已录取的 **V100 / SM70、并发宽度 M=2–16** 算法搬进 FastLLM。不覆盖 MTP verifier 专用路径，也不搬纯 M=1 神优化。

对照源：

- 1Cat 并发设计：`/home/1Cat-vLLM/docs/design/sm70_qwen38_nomtp_concurrency.md`
- 1Cat 集成审计：`/home/1Cat-vLLM/docs/design/sm70_concurrency_integration_audit.md`
- FastLLM SM70 桥：`include/devices/cuda/fastllm-awq-sm70.cuh`
- FastLLM Qwen4-Exp：`src/models/qwen4_exp.cpp`

---

## 1. 目标与非目标

### 1.1 目标

在 4×V100-SXM2-32GB、TP4、Qwen3.8-Flash-Next（NVFP4 或 FP8）上，把 no-MTP decode 并发吞吐拉到 1Cat 已测水平：

| 并发 C | 1Cat 已测聚合吞吐 | 相对单请求 70 tok/s | FastLLM 本方案目标 |
| ---: | ---: | ---: | ---: |
| 4 | 164 tok/s | 58% | ≥160 tok/s |
| 8 | 281 tok/s | 50% | ≥280 tok/s |
| 16 | 441 tok/s | 39% | ≥440 tok/s |

验收口径与 1Cat 对齐：greedy、强制输出 256 token、排除共同 TTFT、独立 8K 级 prompt、checkpoint 原生 KV。不拿 greedy token 哈希当唯一质量门。

### 1.2 非目标

- 不把 FastLLM 改成 vLLM。权重、调度、server 仍走 FastLLM。
- 不搬 Python sidecar、AOTInductor、torch::Tensor 绑定。
- 不搬被 1Cat 拒绝的算法：cross-token expert packing、QSA split-count cap、exact-M5 batched GEMV。
- 不把 MTP / DFlash 作为本方案主路径。MTP verify 的 M=5 只作为 small-N HMMA 的附带收益。
- 不追求 C16 对 70 tok/s 的 65% 效率（728 tok/s）。1Cat 自己也没达到，C16 GPU 已 98% busy。

### 1.3 关键约束

1. **形状门控，不按模型名门控。** 只认 `(M, K, N, group, topk, experts, hidden)`。
2. **CUDA Graph 安全。** 图内禁止 host malloc、CPU 选专家、改 kernel 参数指针。
3. **失败回退现有 TurboMind / GEMV。** 每个新路径都有 `FASTLLM_SM70_*=0`。
4. **权重格式不变。** 继续用 FastLLM 的 `INT4_GROUP` / `FP8_E4M3` / `NVFP4_BLOCK_16`。QPN layout 在 `Prepare*` 时原地改写，不双份常驻。
5. **许可证。** QPN 来自 `v100-skinny`（MIT）。必须保留 `LICENSE.v100-skinny`。Flash-V100 独立模块，不混进 `third_party/turbomind`。

---

## 2. FastLLM 现状与阻塞

并发>1 时，FastLLM 在 SM70 上不是「缺一个 M=1 kernel」，而是 **四层同时卡住**。

### 2.1 模型层：Qwen4-Exp 只能一次跑一个请求

```10159:10161:src/models/qwen4_exp.cpp
        AssertInFastLLM(batch == 1 && inputIds.dims.size() == 2 &&
                        inputIds.dims[0] == 1,
                        "Qwen4-Exp currently runs one request per Forward call.");
```

`--max_batch 8/16` 在 Qwen3.5/3.6 上可用，但 Qwen3.8-Flash-Next / Qwen4-Exp 的 `ForwardBatch` 直接断言 `batch==1`。  
**如果这一层不打开，后面所有 kernel 搬运对并发吞吐都是零。**

同文件里 CUDA Graph 也锁在单 token：

```7956:7960:src/models/qwen4_exp.cpp
        if (!GetFastllmEnv().cudaGraph ||
            hybridMoe ||
            ...
            (hiddenStates.dims[1] != 1 && !mtpTargetGraph) ||
```

以及 TP 可复用 cache：

```1891:1893:src/models/qwen4_exp.cpp
            const bool reusable = Qwen4MtpDraftsPerStep() == 0 &&
                batch == 1 && GetFastllmEnv().cudaGraph &&
```

### 2.2 MoE cache：最多 9 行，图只收 M=1

```1600:1600:include/devices/cuda/fastllm-cuda.cuh
constexpr int FASTLLM_CUDA_MOE_CACHE_MAX_BATCH = 9;
```

`fastllm-moe-cache.cu` 的 `SupportedCacheInput` 同样用这个上限。图捕获还额外要求：

```
hiddenStates.dims[1] == 1 || mtpTargetGraph
```

也就是：eager 最多 9 token，graph 只有 1 token（MTP verify 除外）。C16 直接越界。

### 2.3 Linear 分发：小 M 还在 GEMV / 过晚 repack

FP8 SM70（`src/devices/cuda/linear/fastllm-linear-fp8.cu`）：

| M | 当前路径 |
| ---: | --- |
| 1–4 | 原生 GEMV，**故意不** repack 成 TurboMind |
| 5–31 | 原地 repack 后走 `awq_sm70::GemmFp8` |
| ≥32 | dequant + cuBLAS |

注释写得很清楚：M≤4 原生 GEMV 更快，M≥5 TurboMind MMA 才赢。1Cat 的 QPN8 把这个拐点提前到 M=1，并一直覆盖到 M=32。

NVFP4 SM70：

```
1 ≤ n ≤ 16 且 FastllmCudaGetNcclForceSync()
  → PrepareNvfp4InPlace + GemmNvfp4
否则 native GEMV / dequant
```

并发 decode 的 M=2/4/8/16 能进 TurboMind，但：

- 依赖 `NcclForceSync`，图捕获期可能跳过 repack
- 没有 QPN2/QPN4
- 没有 fused SiLU

AWQ INT4_GROUP：任意 M 都可走 `awq_sm70::Gemm`，但 registry 缺少 N=32/64 的 small-M tactic。

### 2.4 Push all-reduce：骨架在，默认关，且只开 TP2

`src/devices/multicuda/fastllm-custom-allreduce.cu`：

```
FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH   默认 false
wantPush = ok && devices.size() == 2 && CustomArPushEnabledFromEnv()
```

1Cat 的收益在 **TP4、20/40/80 KiB、CUDA Graph**。FastLLM 现在连 TP4 都进不去 push 路径。

### 2.5 已经可复用、不必重写的部分

| 组件 | 位置 | 用途 |
| --- | --- | --- |
| TurboMind s884 桥 | `src/devices/cuda/awq_sm70/` | QPN 失败时的默认 GEMM |
| SM70 registry | `awq_sm70/tm_registry_sm70.cu` | 只注册了 `sm70_884_{4,8,16}` |
| Expert cache | `src/devices/cuda/moe/fastllm-moe-cache.cu` | 图内 refill、pinned host、slot 表 |
| Custom AR | `src/devices/multicuda/fastllm-custom-allreduce.cu` | pull 已默认 auto；push 待开 |
| CUDA Graph pool | `src/devices/cuda/fastllm-cuda.cu` | 捕获/实例化/replay 基础设施 |
| CMake SM70 编译 | `CMakeLists.txt` 429–458 行 | 新 `.cu` 加进 `FASTLLM_CUDA_SOURCES` 即可 |

---

## 3. 1Cat 录取项与搬运优先级

C16 一步约 28.8 ms，GPU 占用 98%，host 间隙 0.54 ms。服务时间拆分：

| 项 | 时间 | 本方案动作 |
| --- | ---: | --- |
| NVFP4 W13 | 5.58 ms | **P0 Direct + fused SwiGLU** |
| Dense GEMM（含 LM-head 签名） | ~8.9 ms | **P0 QPN / P1 small-N** |
| NVFP4 W2 | 2.57 ms | **P0 fused W2** |
| QSA | 1.84 ms | P2 two-warp partial |
| TP | 1.63 ms | **P1 push AR** |
| GDN | 1.35 ms | P2，仅 Qwen3.8 |

### 3.1 必搬（P0 / P1）

| ID | 算法 | 1Cat 源 | FastLLM 接入 | 并发收益 |
| --- | --- | --- | --- | --- |
| A | QPN8 FP8 M=1–32 | `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | Linear FP8 | dense 投影 3–4x |
| B | QPN2/QPN4 NVFP4 M=1–16 | `nvfp4_qpn2_sm70.cu`, `nvfp4_qpn4_sm70.cu` | Linear NVFP4 | down/gate 1.1–1.3x，fused SiLU 关键 |
| C | Direct NVFP4 MoE M=2/4/8/16 | `nvfp4_grouped_decode_sm70.cu` | MoE cache | C16 MoE 主项 |
| D | W13+SwiGLU 融合 M=4/8/16 | 同上 + fused epilogue | MoE | 48 层 0.25–0.54 ms |
| E | Parallel W2 + 定序归约 | 同上 | MoE | 每层数微秒到 9 µs，去掉中间写回 |
| F | E512/K10 router M=1–16 | `csrc/moe/grouped_topk_kernels.cu` 的 SM70 特化 | MoE | 48 层 0.33 ms |
| G | TP4 push all-reduce | 1Cat custom AR；FastLLM 已有骨架 | `fastllm-custom-allreduce.cu` | 20/40/80 KiB 1.8–3.3x |
| H | Small-N HMMA N32/N64 | `operand_sm70_s884.h`, `sm70_884_4.cu` | TurboMind registry | AWQ M=5/8：11.3→9.0 ms |
| I | 固定宽 CUDA Graph | 1Cat FULL graph | `qwen4_exp.cpp` Graph 条件 | C16 host 间隙 <1 ms |

### 3.2 可搬（P2，模型相关）

| ID | 算法 | 条件 | 说明 |
| --- | --- | --- | --- |
| J | Flash-V100 XQA p64 decode | `headDim=256`，E4M3/E5M2 KV | 短上下文不是第一瓶颈；64K+ 才是 |
| K | Prefill BM32 page-784 | 长上下文 prefill | 与并发 decode 正交 |
| L | QSA two-warp partial | Qwen3.8 QSA 12 层 | bitwise，12 层 0.19–0.84 ms |
| M | FlashQLA-SM70 GDN | Qwen3.8 GDN | C16 约 1.4 ms/step |
| N | Shared-expert gate fusion | M=1–16 | 48 层 0.25–0.32 ms |

### 3.3 明确不搬

| 项 | 原因 |
| --- | --- |
| 纯 M=1 QPN 当唯一主路径 | 并发>1 几乎走不到；可作为 M=1 回退 |
| Exact-M5 batched GEMV | 质量不达标，已被 small-N HMMA 替代 |
| Cross-token expert packing | C16 端到端无收益 |
| QSA split-count cap | 改 softmax 结合律，greedy 哈希崩 |
| Marlin | SM70 不可用 |
| FULL 变宽单图 | 1Cat 也没做成默认；改用固定宽多图 |
| Python sidecar / AOTInductor | FastLLM 无此层 |

---

## 4. 目标架构

```
Qwen4Exp::ForwardBatch(batch=C)          # PR0: 打开 C=2/4/8/16
  └─ hidden [1, C, H]
       ├─ Attention / GDN / QSA          # P2
       ├─ Linear dense                   # PR1 QPN
       │     Sm70Dispatch(M,K,N,quant,epilogue)
       │       ├─ FP8  1≤M≤32  → QPN8 (± fused SiLU)
       │       ├─ NVFP4 1≤M≤16 → QPN2/QPN4
       │       ├─ AWQ small-N  → HMMA N32/N64
       │       └─ else         → 现有 TurboMind
       ├─ MoE routed                     # PR2
       │     GPU router E512/K10
       │     Direct W13 (± fused SwiGLU)
       │     Fused W2 + 定序加权归约
       └─ TP all-reduce                  # PR3 push
             SM70 + fully-connected + Graph + 20/40/80 KiB
```

固定宽图：

| 图名 | 覆盖 | 何时 replay |
| --- | --- | --- |
| `decode_m1` | 现有单 token | live=1 |
| `decode_m2` | 2 个 decode token | live=2 |
| `decode_m4` | 4 | live=3–4（pad 到 4） |
| `decode_m8` | 8 | live=5–8 |
| `decode_m16` | 16 | live=9–16 |
| `prefill_chunk` | 现有分块 prefill | 不变 |

live batch 变了只换图，不 recapture。pad 用零路由 / 零 mask，kernel 必须忽略 padding 行的数值贡献。

---

## 5. 新文件与目录

```
src/devices/cuda/sm70/
  sm70_dispatch.cuh              # 形状/环境门控
  sm70_env.cpp                   # FASTLLM_SM70_* 解析
  qpn8_fp8.cu                    # 去 Torch 的 QPN8
  qpn8_fp8.cuh
  qpn2_nvfp4.cu
  qpn4_nvfp4.cu
  nvfp4_moe_direct.cu            # W13/W2 direct + fusion
  awq_moe_grouped.cu             # AWQ compact grouped decode
  router_e512k10.cu              # SM70 E512/K10
  LICENSE.v100-skinny            # 从 1Cat 原样拷贝

include/devices/cuda/
  fastllm-sm70-dispatch.cuh      # 对外 C++ API

test/ops/
  sm70QpnFp8Regression.cu
  sm70QpnNvfp4Regression.cu
  sm70MoeDirectRegression.cu
  sm70PushAllReduceRegression.cu
  sm70DecodeGraphWidthRegression.cpp
```

CMake：把上述 `.cu` 追加到 `FASTLLM_CUDA_SOURCES`（现有 SM70 段，约 433 行）。与 `awq_sm70` 一样，非 sm_70 编译成空实现。

不新建 Python 绑定。测试走现有 `UNIT_TEST=ON` + `ctest -R sm70_`。

---

## 6. 对外 API（建议签名）

全部放在 `fastllm::sm70`，由 Linear / MoE 调用。指针都是已在设备上的 FastLLM buffer。

```cpp
namespace fastllm::sm70 {

enum class Quant { kFp8E4M3, kNvfp4Block16, kAwqInt4Group };
enum class Epilogue { kNone, kBias, kSiluMul, kSwiGlu };

struct DispatchKey {
    Quant quant;
    Epilogue epi;
    int m, k, n;
    int groupSize;     // FP8=128, NVFP4=16, AWQ=32/64/128
    int experts;       // dense=0
    int topk;          // dense=0
};

bool Supported();                   // SM70 && !FASTLLM_SM70=0
bool CanRun(const DispatchKey&);    // 形状门控

// Dense. packed* 是 Prepare* 后的 QPN/TurboMind layout。
bool GemmFp8(const uint8_t* packedW, const half* packedS,
             const half* in, half* out,
             int m, int k, int n, Epilogue epi, cudaStream_t);
bool GemmNvfp4(const uint8_t* storage, const half* in, half* out,
               int m, int k, int n, Epilogue epi, cudaStream_t);

// 权重准备：原地改写，失败不修改源。
bool PrepareFp8Qpn(uint8_t* weight, const float* blockScales,
                   half** packedScales, int k, int n, cudaStream_t);
bool PrepareNvfp4Qpn(uint8_t* storage, size_t bytes, int k, int n,
                     cudaStream_t);

// MoE direct。ids/weights 已在 GPU。
bool MoeDirectNvfp4(const half* in, half* out,
                    const int32_t* ids, const float* routeW,
                    const uint8_t* w13, const uint8_t* w2,
                    int m, int hidden, int inter, int topk,
                    cudaStream_t);

bool RouterE512K10(const half* logits, int32_t* ids, float* weights,
                   int m, int experts, int topk, cudaStream_t);
}
```

`awq_sm70::GemmFp8 / GemmNvfp4` **保持不变**，作为 `CanRun==false` 时的回退。Linear 里先问 `sm70::CanRun`，再问 TurboMind。

---

## 7. 分发规则（必须写进 `sm70_dispatch.cuh`）

统一用 Linear 的 `n`（row / token 数）当 M。

### 7.1 FP8 dense

```
SM70 && FP8_E4M3 && block=128 && K%128==0 && N%32==0
  1 ≤ M ≤ 8    → QPN8 split-K 特化
  9 ≤ M ≤ 16   → QPN8 M16 tile
 17 ≤ M ≤ 32   → QPN8 两阶段 split-K；gated 切成 M16 chunk
  M > 32       → 现有 TurboMind / cuBLAS（不改）
  fused SiLU   → 仅 gate/up 形状 (K, 2N) 且 epi=kSiluMul
```

替换当前 `kSm70Fp8NativeMaxRows=4` 的含义：

- M=1–4 不再锁死 GEMV。QPN8 在这些宽度上应不慢于 GEMV；否则该形状回退 GEMV。
- Repack 阈值从「第一次 M∈[5,31]」改为 **warmup 时对所有 dense 权重做 QPN prepare**。避免图捕获期 `try_to_lock` 失败。

### 7.2 NVFP4 dense

```
SM70 && NVFP4_BLOCK_16 && K%16==0
  1 ≤ M ≤ 16   → QPN2（N 大）或 QPN4（N 中）
  M > 16       → TurboMind（现有 n≤16 限制一并放宽到回退路径）
  去掉对 NcclForceSync 的 prepare 依赖；warmup 阶段完成
```

### 7.3 AWQ dense

```
现有 Int4GroupSm70AwqEnabled 不变
新增：N∈{32,64} 或 M∈{5,8} 时 SelectDispatch 优先 ExactM / ExactMnk
其余继续 Measure/Reuse
```

### 7.4 MoE NVFP4

```
experts==512 && topk==10 && hidden==2560 && inter==160
W13 (K,N)=(2560,320)   W2 (K,N)=(160,2560)
M ∈ {2,4,8,16}         → Direct
  M∈{4,8,16}           → W13 fused SwiGLU
  M∈{2,4,8,16}         → fused W2
M==1                   → 现有 cache fused kernel
M==3,5,6,7,9           → pad 到 4/8/16 再走 Direct；或回退 grouped
其他形状               → 现有 grouped / cache
```

W13 split-K（1Cat 已测，照搬）：

| M | split-K |
| ---: | ---: |
| 2 | 10 |
| 4 | 5 |
| 8 | 4 |
| 16 | 1 |

### 7.5 Push all-reduce

打开条件（全部满足才走 push，否则现有 pull / NCCL）：

```
SM70
devices.size()==4                 # 从 ==2 放宽
peer fully-connected
CUDA Graph capturing or replaying
dtype==FP16
bytes ∈ {20 KiB, 40 KiB, 80 KiB}  # hidden=2560 的 C=4/8/16
FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH 未显式设为 0
```

20/40/80 KiB 对应 `C * 2560 * 2`：C=4/8/16。C=2 是 10 KiB，继续走现有 pull（1Cat 也没把 10 KiB 设为录取点）。

---

## 8. 按 PR 拆分的代码改动

每个 PR 必须：可独立回滚、有算子测试、不改 checkpoint 格式。

### PR0 — Qwen4-Exp 打开 batched decode（1–1.5 周）

**没有这个 PR，后面全是空转。**

改动文件：

- `src/models/qwen4_exp.cpp`
  - 删除 / 放宽 `ForwardBatch` 的 `batch==1` 断言，允许 `1 ≤ batch ≤ 16`
  - `hiddenStates` 从 `[1,1,H]` 扩到 `[1,C,H]`，attention mask / position 按 request 维
  - KV：每个 request 仍持有自己的 paged cache；一步里 C 个 query 走 batched attention
  - CUDA Graph 条件：`hiddenStates.dims[1] ∈ {1,2,4,8,16}`（先只收 2 的幂，其它 pad）
  - TP reusable cache：按 `(batchWidth, capacityClass)` 分槽，不要所有并发请求抢同一块 idle cache
- `include/devices/cuda/fastllm-cuda.cuh`
  - `FASTLLM_CUDA_MOE_CACHE_MAX_BATCH`：9 → 16
- `src/devices/cuda/moe/fastllm-moe-cache.cu`
  - `SupportedCacheInput` 跟上 16
  - Graph 路径允许 `dims[0] ∈ {1,2,4,8,16}`，禁止 hybrid CPU fallback 进图
- 测试：`test/ops/sm70DecodeGraphWidthRegression.cpp`
  - 不跑满模型。用假 Linear + 假 MoE 捕获 `m=2/4/8/16` 四张图，改 metadata 后 replay。

验收：

- `--max_batch 4` 能同时 decode 4 条请求，输出与串行 greedy 在短 prompt 上一致（允许 pad 行全零）
- 无 CUDA Graph 时 eager batched 也正确
- 现有 `batch==1` + Graph 回归不回退

风险：Qwen4 的 QSA / GDN / HyperConnection 大量 `batch==1` 特化。PR0 **先只打通 dense+MoE+attention 的数据布局**，GDN/QSA 可暂时走 per-request 循环，等 P2 再向量化。宁可 C4 eager 先跑通，也不要在 PR0 里重写 GDN。

### PR1 — QPN8 / QPN2 调度骨架（2 周）

源文件去 Torch 化要点：

1. 删除 `#include <torch/all.h>` / `TORCH_LIBRARY`。
2. `torch::Tensor` → `const uint8_t* / const half* / int`。
3. `ATen CUDAGuard` → `FastllmCudaSetDevice` / 调用方已设好 device。
4. 保留 `fp8x8_to_half2x4_fast`、`qpn*_col_from_lane`、prepack kernel。
5. 拷贝 `LICENSE.v100-skinny`。

改动文件：

- 新增 `src/devices/cuda/sm70/*`
- `include/devices/cuda/fastllm-awq-sm70.cuh`：可加薄包装 `GemmFp8(..., Epilogue)`，或让 Linear 直接调 `sm70::`
- `src/devices/cuda/linear/fastllm-linear-fp8.cu`
  - `FastllmCudaHalfMatMulFloatFP8E4M3`：QPN 优先于 `PrepareSm70Layout`
  - warmup：`FastllmCudaWarmupFp8E4M3Sm70` 对 QPN 也 prepare
  - 删除「M≤4 永不 repack」对 QPN 的阻碍；GEMV 仅当 `sm70::CanRun==false`
- `src/devices/cuda/linear/fastllm-linear-fp8.cu` NVFP4 段
  - `FastllmCudaTryNVFP4Sm70TurboMind` 之前插入 QPN
  - 去掉 prepare 对 `NcclForceSync` 的硬依赖
- `CMakeLists.txt` 追加源文件
- 测试：`sm70QpnFp8Regression.cu`、`sm70QpnNvfp4Regression.cu`

算子验收（单 V100，合成 activation，真实或随机权重）：

| 投影 | M | 对照 | 目标 |
| --- | ---: | --- | --- |
| FP8 down | 16 | TurboMind | ≥3x，rel L2 vs FP32 < 1e-3 |
| FP8 gate/up+SiLU | 16 | TurboMind+独立 SiLU | ≥3x |
| FP8 down | 32 | TurboMind | ≥2.5x |
| NVFP4 down | 16 | TurboMind | ≥1.05x，尽量 bitwise |
| 全部 | 1–16 | CUDA Graph replay 5 次 | 与 eager 一致 |

1Cat 参考：FP8 down M16 `185→43 µs`，gate/up+SiLU `356→87 µs`。FastLLM 没有 torch overhead，绝对时间会更好看，相对加速以同进程 TurboMind 为准。

回滚：`FASTLLM_SM70_QPN=0`（总开关）、`FASTLLM_SM70_FP8_QPN8=0`、`FASTLLM_SM70_NVFP4_QPN=0`。

### PR2 — MoE direct batch decode（2 周）

依赖 PR0 的 M≤16 cache 与 GPU 侧 ids。

改动文件：

- `src/devices/cuda/sm70/nvfp4_moe_direct.cu`
  - 从 `nvfp4_grouped_decode_sm70.cu` 移植 `plan_kernel`（按 expert pack，pack=8）
  - Direct W13 / W2，不 sort、不复制 input
  - M=4/8/16 fused SwiGLU epilogue
  - 每 token 10 warp 的 W2 + warp0 定序 FP32 FMA
- `src/devices/cuda/sm70/router_e512k10.cu`
- `src/devices/cuda/moe/fastllm-moe-cache.cu`
  - cache hit 后走 `sm70::MoeDirectNvfp4` 而不是逐专家 Linear
  - Graph 捕获期：`FastllmCudaUseMoeHybrid` 必须为 false
  - miss：现有 pinned refill kernel，不要 host 选专家
- `src/models/qwen4_exp.cpp`：router logits 直接进 GPU top-k，不再下载
- 测试：`sm70MoeDirectRegression.cu`
  - 静态专家 id / 动态 id 两套 Graph replay
  - 对照：逐专家 `GemmNvfp4` + 独立 SwiGLU + 加权和
  - 独立 FP32 oracle，rel L2 < 1e-3
  - 与 grouped 路径不强制 bitwise（TurboMind CTA 顺序会变）

验收：

- 48 层微基准：M=4/8/16 相对当前 cache/Linear 聚合有可见下降（对标 1Cat W13 48 层 1.5–4 ms）
- C8/C16 一步 MoE 墙钟接近 8 ms 量级（V100 TP4）
- `FASTLLM_SM70_MOE_DIRECT=0` 恢复旧路径

AWQ MoE（`awq_moe_grouped.cu`）作为同 PR 的可选后半段：只在 `INT4_GROUP` + 相同 `(E512,K10,H2560,I160)` 时启用。NVFP4 先合。

### PR3 — Small-N registry + TP4 push AR（1 周）

**Small-N**

- `third_party/turbomind/.../operand_sm70_s884.h`：补 `Operand_A_Swizzle_8x64`
- `third_party/turbomind/.../kernel/sm70_884_4.cu`：`ExactMKernelImpl<*,5>`、`ExactMnkKernelImpl`、`Type<8,32,64>` / `Type<8,64,64, GmemLookahead=2>`
- `awq_sm70/fastllm-awq-sm70.cu` `SelectDispatch`：小 N 强制 Reuse 已测 tactic，避免每次 Measure 选到通用 CTA
- 验收：AWQ M=5/8，236 次真实层形状中的代表集，与当前路径 bitwise；墙钟 11.3→9.0 ms 量级

**Push AR**

- `fastllm-custom-allreduce.cu`
  - `wantPush`：`devices.size()==2` 改为 `size==2 || (size==4 && sm70 && fullyConnected)`
  - SM70 TP4 在 Graph 路径、FP16、20/40/80 KiB **默认开**（不再要求 env=1）
  - payload stride 扩大，HC/sum2 信号区放到 payload 之后（1Cat 已踩过图内踩信号）
  - `FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH=0` 仍可关
- 测试：扩 `test/ops/customAllReduceRegression.cpp`
  - TP4、七种消息、七次变值 Graph replay
  - 四卡 bitwise；HC 区 canary 不变

验收：40 KiB push ≥2x vs 现有 pull。

### PR4 — 固定宽 CUDA Graph（1–2 周）

改动：

- `DecodeCudaGraphState` 按 width 存 `segments[width]`
- warmup：对 `max_batch` 向上取 2 的幂，预捕获所有宽度
- live C 不是 2 的幂时 pad；MoE ids 对 pad 行填 `-1`，Direct kernel 已有 `expert>=kExperts` 丢弃
- 禁止 `FastllmCudaMergeMOEUsedGraphUnsafeFallback` 在 SM70 并发图为 true
- 图内只更新 `pinnedDecodeMeta` / page table / expert ids，不改 exec 参数指针

验收：

- C16 GPU busy ≥90%
- host 间隙 <1 ms/step（1Cat 0.54 ms）
- 宽度切换（16→8→4→1→4）不 recapture、数值稳定

### PR5 — Attention / GDN / QSA（按模型，2 周+）

只在 PR1–4 端到端达标后再做。短上下文 C16 里这三项合计约 3.2 ms，不是第一项。

- Flash-V100：建议 `third_party/flash-attn-v100/`，`src/devices/cuda/attention/` 在 SM70 && D=256 时优先于 FlashInfer
- 先 decode XQA p64，再 prefill BM32
- QSA two-warp：保持 split 数，只改 partial warp
- GDN：有 Qwen3.8 再搬 FlashQLA-SM70

---

## 9. 环境变量

默认全部开启（SM70 上按形状自动选）。显式 `0` 回滚。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `FASTLLM_SM70` | on | 总开关 |
| `FASTLLM_SM70_QPN` | on | QPN8/QPN2/QPN4 |
| `FASTLLM_SM70_FP8_QPN8` | on | 仅 FP8 QPN8 |
| `FASTLLM_SM70_NVFP4_QPN` | on | 仅 NVFP4 QPN |
| `FASTLLM_SM70_MOE_DIRECT` | on | MoE direct |
| `FASTLLM_SM70_MOE_FUSED_W13` | on | W13 SwiGLU 融合 |
| `FASTLLM_SM70_MOE_FUSED_W2` | on | W2 定序归约 |
| `FASTLLM_SM70_ROUTER` | on | E512/K10 |
| `FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH` | SM70 TP4 auto-on | 与现有变量兼容；`0` 关闭 |
| `FASTLLM_SM70_DECODE_GRAPH_WIDTHS` | `1,2,4,8,16` | 预捕获宽度 |
| `FASTLLM_DISABLE_SM70_AWQ` | unset | 现有 AWQ 开关，保留 |

不要引入按模型名的 `FASTLLM_SM70_QWEN38=1`。形状不匹配时 `CanRun` 自然返回 false。

---

## 10. 测试矩阵

### 10.1 算子（每 PR 必过）

```
cmake -S . -B build-sm70-tests -DUSE_CUDA=ON -DCUDA_ARCH=70 -DUNIT_TEST=ON
cmake --build build-sm70-tests --target sm70QpnFp8Regression \
    sm70QpnNvfp4Regression sm70MoeDirectRegression \
    sm70PushAllReduceRegression sm70DecodeGraphWidthRegression -j8
ctest --test-dir build-sm70-tests -R '^sm70_' --output-on-failure
```

沿用 `test/ops/README.md` 的 SM70 惯例：缺 GPU 时 exit 77。

每项至少覆盖：

- eager vs Graph replay（改 input / 改 expert id）
- padding 行不影响非 pad 行
- 回滚开关真正回到旧路径
- rel L2 vs FP32（QPN 允许非 bitwise；NVFP4 fused 对 direct bitwise）

### 10.2 端到端

硬件：4×V100-SXM2-32GB，NVLink / fully-connected。

```
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp 4 --cuda_embedding --max_batch 16 \
  --gpu_mem_ratio 0.9 \
  --dtype auto
FASTLLM_CUDA_GRAPH=1
```

对照：

1. 同二进制 `FASTLLM_SM70=0`（旧 TurboMind）
2. 1Cat 同模型同口径数字（上表）

质量：GSM8K 128 题或 PPL；正确率与旧路径差 ≤ 1 题。不要求 greedy 哈希全等。

### 10.3 回归保护

现有必须保持绿：

- `fp8TpRepackDeadlockRegression`（QPN prepare 同样不能在 NCCL 未匹配时拿全局锁）
- `customAllReduceRegression`
- `batch==1` Qwen4 Graph（`decode_m1`）
- MTP / DFlash 路径不强制走新 Direct（`Qwen4MtpDraftsPerStep()!=0` 时 `CanRun` 对 MoE 返回 false，直到单独验收）

---

## 11. 权重准备与显存

1Cat QPN 的关键收益之一是 **内存中性**：prepare 后丢掉 TurboMind 原 layout，约省 6 GiB/rank。

FastLLM 应对齐：

| 格式 | 现在 | 目标 |
| --- | --- | --- |
| FP8 | `PrepareFp8InPlace` 改成 TurboMind pack，scale 新分配 | 再提供 `PrepareFp8Qpn`；warmup 选定一种，**不要 TM+QPN 双驻** |
| NVFP4 | `PrepareNvfp4InPlace` 改成 TM pack | `PrepareNvfp4Qpn` 同样原地；N 不是 32 倍数时沿用现有 pad-or-fail |
| AWQ | 额外 `g_sm70AwqHandles` 持有 TM 缓冲 | 保持；QPN 不做 AWQ dense，AWQ 走 small-N TM |

Prepare 规则：

- 只在 warmup / `FastllmCudaWarmup*` 里做
- 使用现有 `try_to_lock(g_prepareMutex)`，避免 TP 死锁（见 `fp8TpRepackDeadlockRegression`）
- Graph 捕获中 prepare 失败 → 本步回退 GEMV，**不要**在图内 cudaMalloc

Expert cache：

- C16 × topk10 × 48 层不可能把 512 专家全放 GPU。继续 LRU。
- Direct kernel 读 cache slot 指针表（已有）。miss 用现有 pinned refill。
- 并发后 miss 率会升。warmup 用真实路由分布预热；必要时加大 `--moe_cuda_cache`（文档：`docs/cuda-expert-cache.md`）。

---

## 12. 实施顺序与依赖

```
PR0  batched Forward + cache=16 + 假图
  │
  ├─ PR1  QPN dense          ─┐
  ├─ PR2  MoE direct         ─┼─ 可部分并行，但合入顺序 0→1→2
  └─ PR3  small-N + push AR  ─┘   PR3 与 PR1 几乎无文件冲突，可并行开发
           │
          PR4  真固定宽 Graph（依赖 1+2 的 kernel 都 graph-safe）
           │
          PR5  attention/GDN（可选）
```

建议人力：PR1 与 PR3 两人并行；PR2 依赖 PR0 合入后再开。

---

## 13. 风险与对策

| 风险 | 对策 |
| --- | --- |
| Qwen4 `batch==1` 假设散落在 QSA/GDN/PLE | PR0 只保证数据能进 batched Linear/MoE；其余层可先 for-loop C。用 `#ifndef` 或 `if (batch==1)` 保留旧路径 |
| QPN 与 TM 双份权重 OOM | warmup 二选一；QPN prepare 成功则 `IsRepacked=true` 且不再建 TM handle |
| Direct vs grouped 非 bitwise | 独立 FP32 oracle + 端到端分数；不要拿 TM autotune 当金标 |
| Push AR 图内踩信号 | stride 放到 payload 之后；canary 测试 |
| pad 到 4/8/16 浪费算力 | C=3/5/6/7 是过渡态；稳态引擎会填满。kernel 必须对 `ids=-1` 早退 |
| Expert cache 16 行 × 高 miss | 先测；不够再加大 cache bytes，而不是改回 CPU hybrid |
| 许可证 | QPN 文件头保留 MIT 来源；`LICENSE.v100-skinny` 进发行物 |

---

## 14. 验收清单（发布前）

- [ ] PR0：`--max_batch 16` 不再断言；eager C=4 短请求正确
- [ ] PR1：FP8 M16 down ≥3x vs TM；Graph replay 稳定；死锁回归仍过
- [ ] PR2：MoE Direct M=4/8/16 过 FP32 oracle；hybrid 不进图
- [ ] PR3：AWQ M=5 bitwise；TP4 40KiB push ≥2x
- [ ] PR4：C16 busy≥90%，间隙<1 ms
- [ ] 端到端：C4/C8/C16 达到第 1 节目标；GSM8K 不掉质量
- [ ] 所有 `FASTLLM_SM70_*=0` 能回到旧路径
- [ ] 文档：本文件状态改为「已实施」并链到 `docs/benchmarks/` 实测表

---

## 15. 一句话

并发>1 的 V100 收益不在 M=1 GEMV，而在 **先让 Qwen4 能组 batch，再把 M=2–32 的 QPN、MoE direct、small-N 和 TP4 push 塞进固定宽 CUDA Graph**。FastLLM 已有 TurboMind 桥、expert cache 和 custom AR，缺的是形状特化与 `batch>1` 的运行时。按 PR0→PR4 做完，C4–C16 有机会对齐 1Cat；PR5 只在长上下文或 Qwen3.8 细项上加分。

---

## 16. 实施记录

### 16.1 已完成（可在单张 V100 上独立验证）

**PR0 — MoE cache 容量（部分）**

- `include/devices/cuda/fastllm-cuda.cuh`：`FASTLLM_CUDA_MOE_CACHE_MAX_BATCH` 由 `9` 提升到 `16`。
  该常量同时是 `fastllm-moe-cache.cu::SupportedCacheInput` 的行数门控和
  `qwen4_exp.cpp` 的图条件，因此放宽后 eager 路径可直接接受 C≤16 的
  batched decode 输入。它只放宽「接受的行数」，不改变 device cache 的分配，
  于是没有额外显存开销。
- 该改动由既有 `test/basic/test_cuda_moe_cache.cu` 覆盖：测试用
  `FASTLLM_CUDA_MOE_CACHE_MAX_BATCH` 自身构造 `MAX_BATCH` 与 `MAX_BATCH+1`
  两侧边界，常量变化后断言依旧自洽。

**PR1 — QPN8 FP8 内核地基（部分）**

新增文件：

| 文件 | 内容 |
| --- | --- |
| `include/devices/cuda/fastllm-sm70.cuh` | 对外 API：`Fp8QpnSupported / Fp8QpnCanRun / Fp8QpnPrepare / Fp8QpnGemm` |
| `src/devices/cuda/sm70/qpn8_fp8.cu` | 去 Torch 化的 QPN8：prepack、scale 打包、主 kernel、M32 two-phase kernel |
| `src/devices/cuda/sm70/LICENSE.v100-skinny` | 由 1Cat 原样拷贝的 MIT 许可 |
| `test/ops/sm70QpnFp8Regression.cu` | 独立 FP32 oracle 回归 |
| `CMakeLists.txt` | `qpn8_fp8.cu` 加入 `FASTLLM_CUDA_SOURCES`；注册 `sm70QpnFp8Regression` |

去 Torch 化的边界：1Cat 源文件中 `fp8x8_to_half2x4_fast`、`qpn8_*_from_*`
布局映射、`fp8_qpn8_sm70_kernel`、`fp8_qpn8_m32_twophase_sm70_kernel` 与两个
`launch_*` 均为纯 CUDA，**原样保留**；只有 host 层的 `torch::Tensor` 包装被
重写为裸指针。被排除的 `gated_pair` / `ba_split` / `hc_*` 属于 Qwen 专用融合
路径，不在 PR1 范围。

**语义要点（接入时必须知道）**：QPN8 的 FP8 解码器输出的是真值的 `1/256`，
而 `Prepare` 里把 scale 乘了 `256`，两者相消，因此对外可见结果就是普通的
dequant 语义 `out = in @ (fp8_value * scale)`。回归测试正是按这个口径校验的。

验证（本机 4×V100-SXM2-16GB，CUDA 12.8）：

```bash
cmake -S . -B build-sm70-tests -DUSE_CUDA=ON -DCUDA_ARCH=70 -DUNIT_TEST=ON
cmake --build build-sm70-tests --target sm70QpnFp8Regression -j8
CUDA_VISIBLE_DEVICES=0 ./build-sm70-tests/sm70QpnFp8Regression
```

8 个用例全部通过，rel L2 稳定在 `3.4e-4 ~ 4.0e-4`（half 输出量化极限）：

```
[block_m1]        m=1  k=256  n=256   relL2=3.971e-04 OK
[block_m4]        m=4  k=256  n=256   relL2=3.827e-04 OK
[block_m8]        m=8  k=128  n=256   relL2=3.490e-04 OK
[chan_m16]        m=16 k=256  n=256   relL2=3.367e-04 OK
[chan_m32]        m=32 k=256  n=256   relL2=3.535e-04 OK
[chan_m16_k64]    m=16 k=64   n=256   relL2=3.473e-04 OK
[block_m4_hidden] m=4  k=2560 n=2560  relL2=3.522e-04 OK
[chan_m16_down]   m=16 k=5120 n=2560  relL2=3.461e-04 OK
```

覆盖了 block-128 与 channel 两种 scale 布局、M=1/4/8/16/32 三个 kernel 分支、
以及真实 Qwen3.8-Flash-Next 投影形状。回滚开关同样被测：设
`FASTLLM_SM70_FP8_QPN8=0` 后 `Fp8QpnSupported()` 必须为 false。

**PR1 — QPN2 NVFP4 内核地基（部分）**

新增文件：

| 文件 | 内容 |
| --- | --- |
| `include/devices/cuda/fastllm-sm70.cuh` | 追加 `Nvfp4QpnSupported / CanRun / Prepare / Gemm` |
| `src/devices/cuda/sm70/qpn2_nvfp4.cu` | 去 Torch 化的 QPN2：prepack、主 kernel、M=9..16 two-row tile |
| `test/ops/sm70QpnNvfp4Regression.cu` | 独立 FP32 oracle 回归 |
| `CMakeLists.txt` | `qpn2_nvfp4.cu` 加入 `FASTLLM_CUDA_SOURCES`；注册 `sm70QpnNvfp4Regression` |

去 Torch 化边界与 QPN8 相同：`qpn2_col_from_lane`、`qpn2_logical_k`、prepack kernel、
`dequant_e2m1x8`、`nvfp4_qpn2_sm70_kernel` 与 `launch_qpn2` 原样保留；host 层
`torch::Tensor` 改为裸指针。gated SiLU 融合不在本轮（Qwen3.5 dense decode 用不到）。

验证（本机 4×V100-SXM2-16GB，CUDA 12.8）：

```
[m1]        m=1  k=256  n=256   relL2=2.120e-04 OK
[m4]        m=4  k=256  n=256   relL2=2.108e-04 OK
[m8]        m=8  k=128  n=256   relL2=2.086e-04 OK
[m16]       m=16 k=256  n=256   relL2=2.123e-04 OK
[m32]       m=32 k=256  n=256   relL2=2.092e-04 OK
[m4_scale]  m=4  k=256  n=256   relL2=2.108e-04 OK
[m4_k1536]  m=4  k=1536 n=256   relL2=2.082e-04 OK
[m16_k512]  m=16 k=512  n=256   relL2=2.071e-04 OK
```

回滚开关：`FASTLLM_SM70_NVFP4_QPN2=0` 后 `Nvfp4QpnSupported()` 必须为 false。

**端到端（本机已跑，Qwen3.8-27B-QUASAR-NVFP4，TP4，max_batch=4）**

greedy、强制 64 token、短 prompt：

| 并发 C | GEMV/TM 基线 单请求 | GEMV/TM 聚合 | QPN2 单请求 | QPN2 聚合 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.4 | 63.3 | **71.7** | 63.1 |
| 4 | 69.6 | 247.6 | **71.8** | 246.5 |

C=1 约 +5%。C=4 单请求同样略升，短 prompt 聚合被共同 TTFT 稀释。

**官方 `ftllm benchmark`（同模型同二进制，已跑通）**

```bash
PYTHONPATH=/home/fastllm/build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch {1|4} --tokens 4096 \
  --dtype auto --enable_thinking false \
  --input_tokens 64 --output_tokens 64 --batch {1|4} \
  --warmup 1 --temperature 0 --top_k 1
```

| batch | 实际输出 | TTFT | TPOP | Prefill | Batch decode after TTFT | Batch total | Per request |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64/64 | 63.67 ms | 13.80 ms/tok | 1005 tok/s | **72.44 tok/s** | 68.57 | 68.57 |
| 4 | 256/256 | 206.32 ms | 14.08 ms/tok | — | **283.66 tok/s** | 233.83 | 58.54 |

官方口径的 `Batch decode after TTFT` 排除共同 TTFT，C=4 聚合 283.66 tok/s，高于 HTTP 短 prompt 的 247 tok/s。无 conversion error，输出长度打满。

**长 prompt `ftllm benchmark`（greedy，256 out，`--prefix_cache false`）**

```bash
PYTHONPATH=/home/fastllm/build-sm70-tests/tools \
python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --tp 4 --cuda_embedding --max_batch {1|4} --tokens {8192|16384|20480|32768|49152|167936|230400} \
  --dtype auto --enable_thinking false --prefix_cache false \
  --input_tokens {2048|4096|8192|81920|184320} --output_tokens 256 --batch {1|2|4} \
  --warmup {0|1} --temperature 0 --top_k 1
```

| in | batch | 实际输出 | TTFT min–max | Decode after TTFT | Batch total | Per request |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1 | 256/256 | 1.56 s | **71.78 tok/s** | — | — |
| 2048 | 4 | 1024/1024 | 1.57–6.17 s | **123.35 tok/s** | — | — |
| 4096 | 1 | 256/256 | 1.55 s | **71.87 tok/s** | — | — |
| 4096 | 4 | 1024/1024 | 1.57–6.08 s | **123.61 tok/s** | — | — |
| 8192 | 1 | 256/256 | 5.92 s | **70.35 tok/s** | — | — |
| 8192 | 2 | 512/512 | 5.94–11.87 s | **53.58 tok/s** | 33.11 | 16.56 |
| 8192 | 4 | 1024/1024 | 5.92–23.67 s | **47.14 tok/s** | 37.16 | 9.29 |
| 81920 | 1 | 256/256 | 68.91 s | **59.98 tok/s** | 3.50 | 3.50 |
| 81920 | 2 | 512/512 | 69.06–140.19 s | **6.72 tok/s** | 3.53 | 1.77 |
| 184320 | 1 | 256/256 | 188.39 s | **50.13 tok/s** | 1.32 | 1.32 |

C=4 8K 总墙钟 27.56 s。C=2 8K 总墙钟 15.46 s。C=2 80K 总墙钟 145.00 s。8K/80K prefill 基本串行入队（C=2 80K TTFT 69 / 140 s），decode 窗口被最后一条 TTFT 压短。`--max_batch 2 --batch 2` 会卡在 warmup；C=2 需用 `--max_batch 4 --batch 2`。80K 默认 `FASTLLM_PAGED_CUBLAS_CHUNK=8192` 会把 QK workspace 顶到约 1.3GB，16GB 卡上 GPU0 100%/52W 挂死；改 `FASTLLM_PAGED_CUBLAS_CHUNK=2048`、`--tokens 167936`、`--warmup 0` 后跑通。无新 Xid。

**80K C=2 的 6.72 tok/s 不是真实 decode。** 官方 `Batch decode after TTFT` = `(2×255) / (t_end − min(TTFT))`，把第二条 80K prefill（69→140 s）整段算进 decode 窗口。拆开 TPOP：

| 请求 | TTFT | 结束 | TPOP | 含义 |
| ---: | ---: | ---: | ---: | --- |
| #0 | 69.06 s | 145.00 s | 297.77 ms/tok | 出第一个 token 后被 #1 的 80K prefill 堵住 |
| #1 | 140.19 s | 145.00 s | **18.81 ms/tok（53.2 tok/s）** | 两条都 prefill 完之后的稳态 decode |

C=1 80K 已钉死稳态：**Prefill 1188.83 tok/s，Decode after TTFT 59.98 tok/s，TPOP 16.67 ms/tok**。相对 8K C=1 的 70.35 tok/s 只慢约 15%。C=2 第二条的 TPOP 18.81 ms/tok（53.2 tok/s）与 C=1 的 59.98 tok/s 同量级，差在两条共享 80K KV 的 decode 带宽。官方 C=2 的 6.72 tok/s 仍是被第二条 prefill 污染的口径。

C=1 180K（`input_tokens=184320`）同样跑通：Prefill **978.40 tok/s**，Decode after TTFT **50.13 tok/s**，TPOP 19.95 ms/tok。相对 8K 慢约 29%，相对 80K 慢约 16%。KV 上限 230400（1800 pages），AddPrefill 1440 pages 刚好覆盖 180K，加载后单卡约 14.0 GB / 16 GB。无新 Xid。

### 16.2 未完成（需要 32GB V100 + 完整模型才能验证，本轮刻意未盲改）

**PR0 主体：batched `ForwardBatch`**

`qwen4_exp.cpp` 的 `batch==1` 假设不止一处断言，而是散落在
`ForwardTarget`（1000+ 行）、QSA、GDN、HyperConnection、KV append 与
CUDA Graph 的 `graphSequence` 语义里。方案第 8 节估计为 1–1.5 周，且必须用
真实 Qwen3.8-Flash-Next 验证数值。本机为 4×V100-**16GB**（方案假设 32GB），
模型装不下，无法端到端验证，因此不在本轮盲改。

同样未动的还有 `wholeGraphReady` 的 `graphSequence == 1` 条件：把它放宽到
`{1,2,4,8,16}` 会连带改变 KV page append 的 stride、indexer tail 块数、
QSA mask 与 position 假设，风险等级与主体改动相同。

**PR1 接入：`fastllm-linear-fp8.cu` 的 QPN 优先分发**

接入已设计清楚，但**未实施**，原因是它触碰 FP8 权重的生命周期，而这一点在
无模型环境下无法验证。设计与风险如下，供后续实施：

1. `codes` 是 `[K, N]` uint8，与 FastLLM 的 `[N, K]` FP8 `cudaData` 同为
   `N*K` 字节，可**原地**转换：先 prepack 到临时 buffer，再拷回 `cudaData`，
   释放临时 buffer。这样 prepare 后不双份常驻，符合方案第 11 节的内存中性目标。
2. `groupScales`（`(K/128)*(N/32)` 个 half）需要新槽位。现布局是
   `extraCudaData = [scales, bias]`，TurboMind prepare 后会 `push_back(nullptr)`
   变成三槽并以 `back()==nullptr` 作为自己的标记。QPN 可把第三槽设为
   `groupScales`（非空），于是两条路径天然互斥：TurboMind 的 prepare 会因
   `back()!=nullptr` 安全拒绝。
3. **必须同步修改 `FastllmCudaFp8E4M3HasSm70Layout`**：它现在只看
   `IsRepacked && extraCudaData[0] != nullptr`，QPN 准备后会被误判为 TurboMind
   layout 进而用错 GEMM。
4. **风险最高的点**：一旦原地改写 `cudaData`，所有读 FP8 权重的路径都必须
   知道新 layout。当前至少有三条：`FastllmCudaHalfMatMulFloatFP8E4M3`（FP16）、
   `FastllmCudaMatMulFloatFP8E4M3`（FP32，会退回 dequant+cuBLAS 或 GEMV）、
   `FastllmCudaBFloat16MatMulFloatFP8E4M3`（BF16）。漏掉任何一条，权重会被
   当成 `[N, K]` 直接解码，得到**数值错误而非崩溃**，极难定位。接入时必须
   为这三条路径统一加 QPN 分支，并补一条覆盖三种 activation 类型的单测。
5. prepare 时机仍应放在 warmup（`FastllmCudaWarmupFp8E4M3Sm70`），并复用现有
   `try_to_lock` 避免 TP 死锁（回归见 `fp8TpRepackDeadlockRegression`）。

**未搬的内核**：QPN4 NVFP4（`nvfp4_qpn4_sm70.cu`）、MoE direct（PR2）、
small-N / push AR（PR3）。QPN2 内核已落地，并已接入 Linear 分发（待重启服务验证吞吐）。

**PR1 接入：`fastllm-linear-fp8.cu` 的 QPN2 优先分发（已落地）**

接入点：`FastllmCudaHalfMatMulFloatNVFP4Block16` 在 TurboMind / Marlin / GEMV
之前走 `FastllmCudaTryNVFP4Qpn2`。warmup（`NcclForceSync`）期间
`Nvfp4QpnPrepareFromNative` 把 interleaved `[N, (K/16)*12]` 原地改写成
`[codes | raw E4M3 scales]`：native FP32 scale 是 checkpoint E4M3 ×
`weight.scales[0]`（即 `weight_scale_2` / `weight_global_scale`），prepare
按同一 globalScale 反融合回 E4M3，GEMM 再乘一次，对齐 1Cat QPN2 契约。
QPN2 用 `weight.groupCnt = 0x514E32` 标记，避免
把 host sentinel 放进 `extraCudaHalfData`（`cudaFree` 会归到 GPU0，触发
Xid 13/43）。TP shard 会复制 `groupCnt`。M>32 按 32 行切块，decode 与
prefill 共用同一份 packed 权重。FP32 路径走 half adapter；BF16 明确拒绝。
回滚：`FASTLLM_SM70_NVFP4_QPN2=0`（须在权重 prepare 之前设置）。

native unpack 回归（`sm70QpnNvfp4Regression`）：

```
[native_m4]    m=4  k=256  n=256  relL2=2.080e-04 OK
[native_m16]   m=16 k=256  n=256  relL2=2.057e-04 OK
[native_m32]   m=32 k=256  n=256  relL2=2.058e-04 OK
[native_k1536] m=4  k=1536 n=256  relL2=2.081e-04 OK
```

### 16.3 下一步建议

1. C=4 长 prompt 已测完到 8K：2048/4096 聚合约 123 tok/s，8K 因串行 prefill 掉到 47 tok/s，仍低于 1Cat 164 tok/s。下一步看 chunked prefill / 入队是否能并行。
2. QPN8 FP8 按 16.2 接入 Linear（当前模型是 NVFP4，QPN8 需 FP8 权重才能端到端验证）。
3. 在 4×V100-**32GB** 上跑通 Qwen4-Exp `--max_batch 4`（PR0 主体）。
4. 之后按方案第 12 节顺序推进 QPN4 / MoE direct / push AR。
