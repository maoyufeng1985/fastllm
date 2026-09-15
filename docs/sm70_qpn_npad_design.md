# QPN2 侧车 32 列对齐：设计草图

日期：2026-09-14
目标：让 GDN-in（TP4 本地 K=5120，逻辑 N=4120）走 QPN2，取代 TurboMind 回退。
收益实测约 0.57 ms/token（decode 的 4.5%），见 `docs/sm70_ar_microbench.md` §6。

## 1. 用法草图

改完之后，引擎侧两处调用点长这样。

```cpp
// fastllm-linear-fp8.cu, FastllmCudaEnsureNVFP4Qpn2Layout
// 侧车按 32 列对齐后的行数分配；源仍是逻辑 N 行的 native 布局。
const size_t packedBytes = fastllm::sm70::Nvfp4QpnPackedBytes(inputDim, outputDim);
void *packed = FastllmCudaMalloc(packedBytes);
fastllm::sm70::Nvfp4QpnPrepareFromNative(
    (uint8_t *)weight.cudaData, weight.GetBytes(),
    inputDim, outputDim, outputDim,           // k, sourceN, n
    FastllmCudaNVFP4Qpn2GlobalScale(weight),
    cudaStreamPerThread, (uint8_t *)packed, packedBytes);

// 两个取 scales 的位置都改成走同一个 helper
const auto *packedScales = codes + fastllm::sm70::Nvfp4QpnScaleOffset(m, k);

// GEMM 不变，仍写逻辑 N 宽的 output，最后一列块由 kernel 自己屏蔽
fastllm::sm70::Nvfp4QpnGemm(codes, packedScales, cudaInput, cudaOutput,
                            n, m, k, globalScale, cudaStreamPerThread);
```

GDN-in 在 decode 时的实际取值是 `m=5120, k=4120, n=1`，侧车按 4128 行建，
GEMM 只写 4120 列。

## 2. 签名

`include/devices/cuda/fastllm-sm70.cuh`

```cpp
// 侧车必须容纳的行数：输出列按 32 列 tile 向上对齐。
inline int Nvfp4QpnPackedRows(int n);
inline size_t Nvfp4QpnPackedBytes(int k, int n);
// 侧车里 packed scales 相对 codes 起点的字节偏移。
inline size_t Nvfp4QpnScaleOffset(int k, int n);

// 形状门放宽：允许 n 不是 32 的倍数，其余不变。
bool Nvfp4QpnCanRun(int m, int k, int n);

// storage 是 native [sourceN, (k/16)*12] 交错布局。侧车按
// Nvfp4QpnPackedRows(n) 行建立；列号 >= sourceN 的位置写 0，保证最后一列
// tile 读到的字节是定义过的。
bool Nvfp4QpnPrepareFromNative(uint8_t *storage, size_t storageBytes,
                               int k, int sourceN, int n, float globalScale,
                               cudaStream_t stream, uint8_t *dest = nullptr,
                               size_t destBytes = 0);

// 不变。要求 n % 32 == 0；带 padding 的侧车只走上面那个入口。
bool Nvfp4QpnPrepare(const uint8_t *weight, const uint8_t *scales,
                     uint8_t *codes, uint8_t *packedScales,
                     int k, int n, cudaStream_t stream);
```

`src/devices/cuda/sm70/qpn2_nvfp4.cu`

```cpp
// 签名不变，n 仍是逻辑输出列数，也仍是 output 的行跨度。
template <int SplitK, int NAcc, int RowTiles>
__global__ void nvfp4_qpn2_sm70_kernel(const uint8_t *codes,
                                       const uint8_t *group_scales,
                                       const half *input, half *output,
                                       int n, int k, int m, float global_scale);

// 两处 prepack 各加一个 sourceN，列号超界的 lane 写 0 而不是去读源。
__global__ void nvfp4_qpn2_prepack_codes_from_native_kernel(
    uint8_t *output, const uint8_t *source, int n, int source_n, int k,
    int source_row_bytes);
__global__ void nvfp4_qpn2_prepack_scales_from_native_kernel(
    uint8_t *output, const uint8_t *source, int n, int source_n, int k,
    int source_row_bytes, float global_scale);

// grid.x 从 n/32 改成 (n + 31)/32。
template <int SplitK, int NAcc, int RowTiles>
void launch_qpn2(...);
```

## 3. 数据形状与所有权

| 对象 | 形状 | 谁拥有 |
|---|---|---|
| `weight.cudaData` | native `[sourceN, (k/16)*12]`，N=4120 | Data，不变 |
| `weight.nvfp4Qpn2Packed` | `[codes (P*k/2) \| scales (P*k/16)]`，P = align32(4120) = 4128 | Data，不变 |
| `cudaOutput` | `tokens × 4120` | 调用方，不变 |

侧车的 tile 数与 GEMM 的 grid.x 都取自 P；输出的行跨度与 store 上界取自逻辑
n。两者只差一个 `Nvfp4QpnPackedRows`。

`nvfp4Qpn2Packed` 是独立分配，不动 `cudaData`，所以 native 布局留给 prefill
的既有行为完全不变。GDN-in 的 prefill 会从 TurboMind 换到 native 反量化加
cuBLAS，和另外 208 条 QPN2 投影现在的走法一致。

## 4. 不变式

1. `n % 32 == 0` 时，`Nvfp4QpnPackedRows(n) == n`，侧车字节数、grid、
   store 上界、scales 偏移全部和改动前逐位相同。208 条既有投影的 token 流
   必须不变。
2. `Nvfp4QpnPrepareFromNative` 写完的侧车，列 `[0, sourceN)` 有真实数据，
   列 `[sourceN, P)` 全 0。kernel 对第 `P/32 - 1` 个 tile 仍会读这些字节，
   E2M1 nibble 0 与 E4M3 0 都解码成 0。
3. store 的屏蔽条件是 `tile * 32 + output_col < n`，`tile` 来自 blockIdx.x，
   `output_col` 来自 `element & 31`。列与列之间在 accumulator、shared
   partials、epilogue 上都不相交，所以屏蔽最后一列 tile 不影响其它列。
4. 侧车一旦建立，`weight.IsRepacked` 不会再被 TurboMind 置上，因为 QPN2 在
   分发表里排在 TurboMind 前面，且 GDN-in 的形状门现在能过。

## 5. 风险

**显存。** 侧车字节是 `(9/16) * P * K`，P 是 32 对齐后的行数。逐条算：

| 投影 | 数量/rank | 单条字节 | 小计 |
|---|---:|---:|---:|
| N=5120, K=5120 | 128 | 14,745,600 | 1,887,436,800 |
| N=8704, K=5120 | 64 | 25,067,520 | 1,604,321,280 |
| N=3584, K=5120 | 16 | 10,321,920 | 165,150,720 |
| **既有合计** | 208 | | **3.66 GB** |
| GDN-in 4120→4128 | 48 | 11,888,640 | 570,654,720 |
| **改动后合计** | 256 | | **4.23 GB** |

之前草稿写的「既有约 2.9 GB」偏低约 26%，以 3.66 GB 为准。
侧车走的是 `FastllmCudaMalloc` 而不是 `FastllmCudaMallocModelWeight`，绕开了
权重 slab 的预留，也不进加载期的显存估算。这一条是既有行为，改动只是把它
放大。

**实测（2026-09-15，80K C=1，QPN2 缺省开，逐秒 `nvidia-smi` 采样 88 点，
`/tmp/v3_mem_samples.txt`）：**

| 量 | GPU0 | GPU1-3 | 对 16384 MiB 的余量 |
|---|---:|---:|---:|
| 稳态平台（t=40 到 85s，占绝大多数时间） | 14949 MiB | 14871 MiB | **1435 MiB（1.40 GiB）** |
| 瞬时峰值（t=53 到 56s，warmup/capture 窗口） | **16113 MiB** | 16035 MiB | **271 MiB（0.26 GiB）** |

四卡同步升降，确认不是共租户。结论：**稳态余量够，但 warmup/capture 的瞬时
余量只有 271 MiB**，所以这张 16 GB 卡上不能叠 C=4 长 prompt，任何额外显存
开销都可能 OOM（本轮 V2 的 C=4 尝试正是这样 OOM 的）。上面的「要靠重启后的
`nvidia-smi` 实测确认」到此关闭。

**prefill 路径变化。** GDN-in 从 TurboMind W4A16 换成 native 反量化加
cuBLAS。这是既有 208 条投影的常态，但 GDN-in 是 48 层的最大一条，要单独量
prefill 的 tok/s 前后对比。

**实测（2026-09-15，当前二进制，同形状 QPN2 on/off 对照）：**

| 场景 | QPN2 off（TurboMind GDN-in） | QPN2 on（native + cuBLAS） | 差 |
|---|---:|---:|---:|
| 8K C=1 prefill | 2570.30 tok/s | 2575.25 tok/s | +0.19% |
| 80K C=1 prefill | 1993.36 tok/s | 1988.78 tok/s | −0.23% |

即 prefill **不掉**（8K 略增、80K 略减，都在 0.25% 内），两个形状的 greedy
sha256 都一样（8K C=1 `02c702ca`）。上面的「要单独量 prefill 前后对比」到此关闭。

**M=9..16 的 two-row tile。** 走的是同一个 epilogue，屏蔽条件复用，测试已覆盖
（`native_pad_m16`、`split_pad_m16`）。

## 6. 对抗复查后的加固

一轮独立复查按上面这份草图逐条对代码，结论是设计在 4120 这条路径上算得对，
问题集中在失败路径和契约。已落地的三处：

**侧车建立失败不再永久降级。** 原来 `FastllmCudaEnsureNVFP4Qpn2Layout` 失败会
直接落到 `FastllmCudaTryNVFP4Sm70TurboMind`，而它在 SM70 上 forceSync 恒真，
会走 `PrepareNvfp4InPlace` 原地改写 `cudaData` 并置 `IsRepacked = true`，此后
QPN2 再也建不起来，没有日志也没有重试。新增 `Data::nvfp4Qpn2Wanted`：形状门
一过就置上，`FastllmCudaEnsureNVFP4Sm70TurboMindLayout` 见到它就拒绝，失败
于是退到 native 反量化这条非破坏性路径。分配失败与转换失败各打一行日志。

**抓图期不再尝试建立。** `Nvfp4QpnPrepareFromNative` 里有真实 `cudaMalloc` 和
结尾的 `cudaStreamSynchronize`，在 capture 里调用会失败或破坏 capture。
`FastllmCudaTryNVFP4Qpn2` 现在用 `FastllmCudaGraphIsCapturingFast()` 提前返回。

**侧车改为预热期提前建立。** `Qwen3_5Model::OnAutoWarmupFinished` 里照既有
FP8 布局的做法遍历一遍权重，对每条合格 NVFP4 权重调用
`FastllmCudaWarmupNvfp4Qpn2Sm70`，在空闲且无 capture 时把 256 条侧车一次建完。
懒建立只剩兜底，不再是主路径。

**GEMM 的生产者契约写清楚。** `Nvfp4QpnGemm` 现在写明侧车必须按
`Nvfp4QpnPackedBytes(k, n)` 分配，且 `Nvfp4QpnScaleOffset` 要用同一组 `(k, n)`。

## 7. 测试

`test/ops/sm70QpnNvfp4Regression.cu` 现在 20 个用例：

- 既有 14 个（`n=256` 的未对齐前路径）保持不动，relL2 与改动前逐位相同，
  这是不变式 1 的回归。
- 新增 3 个非 32 倍数用例，含 `k=5120, n=4120` 的真实 GDN-in 形状，输出尾部
  32 个 half 放 canary。
- 新增 3 个 split 用例，用引擎那套分配切分（`storage` 只给 native 大小，
  侧车单独 `Nvfp4QpnPackedBytes`）。`split_pad_gdn` 与 `native_pad_gdn` 的
  relL2 都是 2.103e-04，说明原位与分离两条路一致。

canary 做过反向对照：把 store 屏蔽条件改回 `output_row < m` 后，
`native_pad_gdn` 报 canary=8 FAIL，`native_pad_m16` 报 canary=28 且 relL2 从
2.067e-04 涨到 1.662e-01。工具确实能抓到越界，不是空过。

## 8. 实测结果

工作负载：Qwen3.8-27B-QUASAR-NVFP4，TP4，`--input_tokens 1024 --output_tokens 128
--batch 1 --warmup 1`，greedy，CUDA Graph 开。基线树只回退本文这 6 个文件，AR
那套策略改动两边都在。

| 指标 | 基线 | 本改动 | 差 |
|---|---:|---:|---:|
| decode TPOP | 13.045 ms（13.06 / 13.03） | 12.900 ms（12.88 / 12.90 / 12.90） | **−0.145 ms，+1.11%** |
| decode tok/s | 76.65 | 77.54 | +1.16% |
| prefill tok/s | 2329.6 | 2296.3 | **−1.43%** |
| GPU0 free（AutoWarmup 时） | 3.55 GB | 4.38 GB | +0.83 GB |
| token stream sha256 | `539186eb…` ×3 | `539186eb…` ×2，`d5fcc5fc…` ×1 | 见下 |

基线有一次运行异常（TPOP 94.98 ms、prefill 996.9 tok/s），已剔除；两侧其余
样本的离散度都在 0.2% 以内。

### 8.1 为什么只有 1.1%，而不是预估的 4.5%

预估把 QPN2 在 N=4128 上按 N=5120 的效率缩放，得到约 16.3 µs/次。引擎内实测
不是这样：

| 路径 | 每次调用（引擎内 p50） |
|---|---:|
| 基线：TurboMind `gemm_kernel` grid=(2,17,7) | 30.78 µs |
| 基线：`CropNvfp4OutputKernel` grid=(17,1,1) | 1.44 µs |
| **基线合计** | **32.2 µs** |
| 本改动：`nvfp4_qpn2_sm70_kernel` grid=129 | 26.66 µs |

QPN2 在 N=4128 上只比 N=5120 便宜一点点（grid=160 的 p50 是 23.94 到 24.19 µs），
**这个形状在 M=1 是延迟受限，不是按列数受限**。所以每次只省 5.5 µs，48 层
264 µs 的 kernel 时间，落到墙钟是 145 µs，剩下的被通信重叠吃掉。

同时确认了不变式 1 在引擎层面成立：grid=160 那族两边都是 16896 次调用，
p50 24.19 对 23.94 µs，本改动没有扰动既有 208 条投影。

### 8.2 prefill 退化的来源

GDN-in 的 prefill 从 TurboMind W4A16 换成了 native 反量化加 cutlass：

| kernel | 基线 | 本改动 | 差 |
|---|---:|---:|---:|
| `FastllmCudaNVFP4Block162HalfKernel` | 6656 次 / 1111 ms | 8192 次 / 1339 ms | +1536 次，+228 ms |
| `cutlass_70_tensorop_h884gemm_128x128` | 6144 次 / 6165 ms | 7680 次 / 7626 ms | +1536 次，+1461 ms |

### 8.3 显存方向与预估相反

侧车确实多了 570 MB/rank，但**基线的 TurboMind 回退为了 crop 掉 8 列，会常驻
一块按服务 token 预算分配的 scratch**：

```
scratch = maxTokens × packedN × 2 = 167936 × 4128 × 2 = 1.291 GiB
侧车    = 48 × 11,888,640 B                        = 0.531 GiB
净释放  = 0.760 GiB = 0.816 GB        实测 +0.83 GB
```

注意这依赖 `--tokens`。交叉点在 `maxTokens ≈ 67,400`：`--tokens` 低于这个值时，
侧车比 scratch 大，本改动反而多占显存；高于这个值才释放。

### 8.4 结论与建议

**建议不落地。** 实测是一笔很薄的交易：decode +1.1%，prefill −1.45%，显存只在
大 `--tokens` 下更好，代价是改掉 QPN2 的 kernel 契约加约 120 行跨 6 个文件。
原计划里这一项排在第二优先，依据的是「反量化 + FP16 GEMM 占 27%–29%」那个
prefill 污染的占比；按实测它不配那个位置。

如果仍要拿这 1.1%，正确的做法是补上 8.2：让 GDN-in 的 prefill 留在 TurboMind。
侧车是独立分配，`cudaData` 之后再被原地重打包不会破坏它，所以可以在侧车建好
之后放行 TurboMind 的大 M 重打包。那需要放开 `FastllmCudaEnsureNVFP4Sm70TurboMindLayout`
的 `n <= 16` 门，是另一件事，必须单独量。

**token 一致性没有保证。** 3 次本改动运行里 2 次与基线逐位相同，1 次不同
（`d5fcc5fc…`）。基线 3 次都是同一个哈希。样本太少，不能断定这 1/3 是本次
改动引入的，但也不能声称逐位一致：GDN 是带状态的递归，末位差会被放大。要
落地必须先过质量门。

## 9. 仍未做

- 质量门（GSM8K 一类）。1Cat 用的是 250 题。
- `Data::Allocate`/`Resize` 重新分配 `cudaData` 时没有清掉侧车（既有问题）。
- `Nvfp4QpnPackedBytes(k, n)` / `Nvfp4QpnScaleOffset(k, n)` 的参数是
  (归约维, 输出维)，而 `fastllm-linear-fp8.cu` 里的局部名是 (m=归约, k=输出)，
  两种命名在同一处相遇，容易看错。


