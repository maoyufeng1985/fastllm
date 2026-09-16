# attention 桶优化方案（2026-09-16）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-16GB / TP4 / no-MTP / 180K prompt + FP8 KV。
前提：`sm70_4x200k_rotate_plan.md` 已把 D256 workspace 从 GEMM 路线里划走，指出它的靶子是
attention 桶（12.4%），不是 GEMM 桶（46.7%）。本方案接住那个靶子，回答三件事：attention 桶
里到底是多少、能挤出来多少、1Cat 的 D256 设计能不能搬。

## 0. 一句话结论

**attention 桶的账是真的，但它是个小桶，而且这个内核算的不是数学，是"同一行读三遍 + 把
25% 的行补零写一遍"。**

实测（`/home/tools/attn_softmax_recommend.cu`，V100 锁频 1530 MHz，真实几何
`rows=4096 ch=8192 base=4096`，复刻耗时 0.2467 ms vs trace 250.9 µs，复刻对得上）：

| 变体 | 耗时 | 相对现役 | 成立条件 |
|---|---:|---:|---|
| 现役结构 | 0.2467 ms | — | — |
| 在线单读（读 2 遍代替 3 遍） | 0.2271 ms | **−8.2%** | 无条件，数值等价 |
| 不补零（少写 25%） | 0.2096 ms | **−15.0%** | 需要 PV 侧不再读被 mask 的列 |
| 在线单读 + 不补零 | 0.1966 ms | **−20.5%** | 同上 + 无条件 |
| 纯拷贝地板（读写各一遍可见区） | 0.1408 ms | −43.0% | 理论地板，不可达 |

换算到 prefill 墙钟（干净跑 180K prefill ≈ 83 s，本路径单流无重叠，kernel busy 与墙钟近似
1:1）：**Step 1 无条件能拿 −1.2%，加上「不补零」这条能拿 −3.0%，把 softmax 融进 QK GEMM
（Step 3）才是唯一有量的，上限 −14.8%、现实 −8~11%。**

结论排序：**Step 3 融 GEMM 是主菜，Step 1 是顺手能捡的，D256 workspace 本身不要搬。**

## 1. 两个桶的账，先把归属钉死

180K prefill 实测（`/home/nsys/pf180k.sqlite`，180K prompt + chunk 4096 + 200K 池 + FP8 KV +
low_gpu_mem，device 0，窗口 118.20 s、kernel busy 108.45 s = 91.8%）：

| 内核族 | 每卡时长 | 占 busy | 说明 |
|---|---:|---:|---|
| cutlass Kernel2（NVFP4 GEMM） | 39.160 s | 36.1% | 253696 次发射 |
| NCCL AllReduce RING_LL | 36.099 s | 33.3% | PCIe 结构性成本 |
| **`FastllmPagedCublasSoftmaxWithCausalMask`** | **12.278 s** | **11.3%** | **attention 桶的主体，49056 次发射** |
| volta_s884gemm_fp16 | 9.076 s | 8.4% | QK/PV 的 cuBLAS 就算在这一族里 |
| volta_h884gemm_64x64 | 2.931 s | 2.7% | |
| `FastllmCudaNVFP4Block162HalfKernel`（反量化） | 2.070 s | 1.9% | |
| `FastllmPagedCublasAttnBlockUpdateFloat` | 0.641 s | 0.6% | 在线 softmax 的 running max/sum 合并 |
| `FastllmPagedCacheGatherHeadRangeKernel`（FP8 KV 解量化搬移） | 0.401 s | 0.4% | |
| `FastllmPagedAttentionSplitTiledGqaKernel` | 0.050 s | 0.05% | decode 侧残留，prefill 不走 |

三点记账说明：

1. **attention 桶 13.320 s = softmax 12.278 + update 0.641 + gather 0.401，占 busy 的
   12.28%**（对得上旧口径 12.4%），softmax 一个内核占桶的 92%。
2. **QK/PV 的 GEMM 不在这个桶里**。prefill 的 QK/PV 走 `cublasHgemm`，在 trace 里落进
   "FP16 GEMM 三族"那个桶，无法从 trace 单独拆出 attention 的部分。含 QK/PV 约 20% 属估算。
   做内核优化按 20% 估天花板，算 Step 1/2/3 的直接受益按 12.28% 的 softmax 桶算，别混。
3. **GEMM 口径对照**：Kernel2 36.1% + FP16 GEMM 三族约 11.6% = 47.6%（本方案复算）。旧文档
   写 46.7%，差在 FP16 三族的取法，两者都可接受，不以此为准。

模型几何（`/home/models/Qwen3.8-27B-QUASAR-NVFP4/config.json`）：64 层只有 16 层是
`full_attention`，48 层是 `linear_attention`；`head_dim=256`、`num_attention_heads=24`、
`num_key_value_heads=4`，TP4 下每卡 1 个 KV head、6 个 Q head（`group=6`）。这解释了
attention 为什么只占 12.4%：只有 1/4 的层在做 O(context²) 的活。

## 2. 从 trace 反推真实几何（这一步定死了后面所有测量）

调用点：`FastllmPagedCublasSoftmaxWithCausalMask<<<qoLen,256>>>(qk,qk,qoLen,chunkLen,
kvLen-qoLen-kvStart, ...)`（`fastllm-paged-attention-native.cu:1045`），所以
`channels = chunkLen`、`base = kvLen - qoLen - kvStart`。

分块 prefill 里 `kvStart = ci*chunkLen`、`kvLen = (ci+1)*chunkLen`，代入得

**`base = chunkLen - qoLen = 8192 - 4096 = 4096`，对每一个 chunk 都是同一个值。**

这一条解释了 trace 里最反常的现象：

| gridX | 发射次数 | 总时长 | 单次 |
|---:|---:|---:|---:|
| 4096 | 46848 | 11.755 s | 250.9 µs |
| 3872 | 2112 | 0.523 s | 247.6 µs |
| 16 | 96 | 0.000 s | 3.5 µs |

46848 次发射耗时全都一样（直方图 260 µs 档占 38906 次），**不是因为"整行完全可见"，而是
因为每次发射的可见分布完全相同**：第 o 行看到 `o+4097` 列（上限 8192），即 4097..8192 列，
均值 6144 = chunk 的 75%。每行还有 2048 列（25%）被 mask 掉，由补零那趟写掉。

复刻验证：`base = chunkLen - qoLen = 4096` 时耗时 0.2467 ms，对 trace 的 250.9 µs 误差 −1.7%；
换成"整行完全可见"（`base = chunkLen-1`）复刻耗时 0.3006 ms，偏 +19.8%，对不上。**所以
trace 的几何是三角那一档，补零那趟是真的在写数据，不是空转。**

由此得到全部流量的账（按发射几何精确累加，非估算）：每次发射动 `4096 × 8192 × 2 B = 67.1 MB`，
46848 次合计 **3.28 TB** 的 P 矩阵；softmax 读 3 遍 × 75% + 写 1 遍 × 75% + 补零 1 遍 × 25%
= 3.25 个整 chunk 的 pass，共 **10.65 TB** 逻辑流量，12.278 s → 有效 **0.87 TB/s**。这已经
**顶到本机实测的拷贝屋顶 791 GB/s 之上**（`/home/tools/dram_bw`：读 893、写 840、拷贝
791 GB/s），多出来的是 L2 命中。换句话说：这个内核没有漏掉可捡的带宽，它只是把同一份数据
搬了太多遍。

## 3. 机制判定：四路交叉都指向"不是数学"

复刻与全部变体在 `/home/tools/`：`attn_softmax_bench.cu`、`attn_softmax_probe.cu`、
`attn_softmax_routes.cu`、`attn_softmax_online.cu`、`attn_softmax_recommend.cu`。
V100 锁频 1530 MHz（`nvidia-smi -lgc 1530,1530`），`rows=4096 ch=8192 reps=40`，可复跑。

| 变体 | 耗时 | 相对现役 | 判定 |
|---|---:|---:|---|
| 现役结构（3 读 + 1 写 + 补零） | 0.2467 ms | — | 基线，对齐 trace |
| 同结构，`expf` 换成 2 个 FLOP | 0.2456 ms | −1.2% | **数学不是瓶颈** |
| `exp2f` 替代 `expf` | 0.2458 ms | −1.1% | 同上，路口关掉 |
| half2 / half4 向量化 | 0.2449 ms | −1.5% | 访存宽度不是瓶颈 |
| warp-per-row（去掉块级归约） | 0.3643 ms | +46% | 块级归约不是瓶颈 |
| 共享内存暂存（DRAM 读一遍） | 0.2398 ms | −3.5% | 不划算，smem 挤 occupancy |
| 在线单读（读 2 遍） | 0.2271 ms | −8.2% | 有效 |
| 不补零 | 0.2096 ms | −15.0% | 有效，但有前提（见 §5 Step 2） |
| 在线单读 + 不补零 | 0.1966 ms | −20.5% | 两个都成立时的组合 |
| 纯拷贝地板（读写各一遍可见区） | 0.1408 ms | −43.0% | 理论地板 |

数值等价性（同一输入，与现役内核逐元素比，只比可见区）：

| 变体 | 最大 half 偏差 | 超 1e-3 的元素数 | running sum 相对误差 |
|---|---:|---:|---:|
| 在线单读 | 0.000001 | 0 | 2.84e-07 |
| 在线单读 + 不补零 | 0.000001 | 0 | 2.84e-07 |

**这两行是 Step 1 能直接落地的前提：省掉一趟读之后，数值和现役内核是等价的。**

## 4. 为什么和旧结论"瓶颈是 softmax 数学吞吐"不矛盾

是两次量了不同的东西。

旧结论（`sm70_1cat_port_plan.md:284-301`、`sm70_status_and_backlog.md:292-299`）测的是
**decode** 侧同几何探针：读天花板 786 GB/s = 88% 屋顶，加上真实在线 softmax 后掉到
456 GB/s = 51%，于是判定"瓶颈是 softmax 数学吞吐"。那个判定对 decode 成立：S=192，行很短，
归约与 `exp` 的开销盖过访存。

本方案测的是 **prefill** 侧的实际内核，四路交叉（去掉 `exp`、换 `exp2f`、向量化、去掉 warp
归约）全部不指向数学，且几何完全不同（行宽 8192、每次发射 4096 行、可见区恒定 75%）。

**结论：prefill 的 softmax 是"趟数瓶颈"，decode 的是"数学瓶颈"，两者的结论不能互相引用。**
旧文档里"没有白捡的内核优化"那句对 prefill 不成立，本方案 §3 的表就是反例。

## 5. 分步方案

### Step 1. 在线单读 softmax（无条件，先拿）

- 改什么：`FastllmPagedCublasSoftmaxCausalFunc`（`fastllm-paged-attention-native.cu:355-415`）
  现在是 4 趟（367 求 max / 387 求 sum / 410 归一化写出 / 412 补零），改成 2 趟：第一趟每线程
  跑在线 (m, s)、块级用带 rescale 的归约合并，第二趟再读一遍、归一化、写出。补零那趟保留。
- 额外好处：在线版把 running-state 的合并收进同一趟，减少一次跨块依赖。
- 开关：新增 `FASTLLM_PAGED_SOFTMAX_ONLINE`，默认关，A/B 用，现役结构保持可回归。
- 验证门：(a) `bits_diff=0`（可见区逐元素 ≤1e-3），(b) prefill tok/s ≥ +3%，(c) 4×200K 场景
  exit 0 无 OOM，(d) 极端分布用例（单行内量级跨 20）不退化。
- 风险：在线版在 `x > running_max` 时多算一次 `expf`，随机分布下实测无退化，病态分布需补测。
  动的是所有 prefill 请求都走的公共内核，必须有开关兜底。

### Step 2. 不补零（−15%，有前提）

- 现在为什么补零：softmax 把被 mask 的列写成 0（`:412-414`），PV GEMM 才能整块读 P 而不必
  逐行限制列范围。省掉这趟写，就必须让 PV 侧改成"按行只读 `visible` 列"，或者把 mask 移到
  PV 的 epilogue 里用条件替代（1Cat `prefill.cu` 的做派：prefix 段无 mask、tail 段精确因果，
  P 里根本不产生被 mask 的列）。
- 收益：内核 −15%；与 Step 1 组合后 −20.5%。换算 prefill 约 **−2.2%**（单独）到 **−3.0%**（组合）。
- 验证门：与默认逐元素比，可见区必须完全一致；确认 PV 没有把未初始化的列吃进输出。
- 风险：改的是 PV 的读取语义，回归面比 Step 1 大。排在 Step 1 之后单独 A/B。

### Step 3. softmax 融进 QK GEMM epilogue（有量，主菜）

- 为什么是主菜：现在 P 矩阵要走"QK 写一遍 → softmax 读 3 遍写 1 遍 → PV GEMM 再读一遍"，
  10.65 TB 显存流量换一个中间结果。把这个中间结果留在寄存器/smem 里，softmax 的 12.278 s
  就基本消失。
- 怎么落地：现在是 `cublasHgemm` 直调（`:944`、`:1027`），cuBLAS 不给 epilogue 挂钩，所以要
  自写一个 sm70 的 QK GEMM 带 softmax epilogue。这正是 1Cat
  `csrc/attention/sm70_v37/gemm_with_softmax.h` 的做法（M128/N256/K32 tile + softmax visitor +
  同流上的一次 final-reduction），是那套设计里唯一真正值得搬的东西。
- 分两步，别一次吞完：
  - 3a 只融 softmax：QK epilogue 里算 `exp`、写归一化后的 P，P 仍写一次给 PV 读。预期
    softmax 12.278 s → 3~5 s（剩写 P 的代价），prefill **−8~11%**。
  - 3b 再融 PV：P 完全不落 DRAM，上限 −14.8%（等于这段内核全免），但寄存器/smem 吃紧。
- 验证门：fp16/fp8 两条 KV 路径各跑 180K A/B，要求 `bits_diff=0`、prefill ≥ +6%，且 8K/80K
  短上下文不回退（短上下文下融合收益变小，别做成倒退）。
- 风险：自写 sm70 GEMM 是本方案最大工程量；cutlass 在 7.0 上要避开 tf32/bf16 的 tensor-core
  假设。收益取决于 P 还得 round-trip 几次（3a 拿不满）。

### Step 4.（可选）并掉 update 内核

`FastllmPagedCublasAttnBlockUpdateFloat` 只有 0.641 s（0.6%）。若 Step 3 的 epilogue 里已经
持有 running max/sum，这个内核的前提就没了，跟着删。单独做不值得。

## 6. 1Cat 的 D256 设计，哪些能搬、哪些不能

读的是 `/home/1Cat-vLLM/csrc/attention/sm70_v37/`（`prefill.cu` 666 行、`tail.cu`、
`gemm_with_softmax.h`、`README.md`）。它由三块组成，逐块判断：

| 组成 | 它做什么 | 能不能搬 |
|---|---|---|
| QK GEMM 带 softmax epilogue（`gemm_with_softmax.h`） | QK 算完直接算 softmax，P 不落显存 | **能，且值得**。就是 Step 3，这套设计里唯一有量的一块 |
| prefix/tail 因果切分（`prefill.cu:481-487`） | `prefix = total_kv - kTail` 后，prefix 无 mask、tail 走标准 FA2 | **能，是 Step 2 的刀**：让 P 里根本不产生被 mask 的列，补零那趟自然消失 |
| 768 MiB score 缓存 + tail stream（`prefill.cu:382-418`、`:560-661`） | 缓存 49152×8192 的 fp16 score 张量，用私有流把 causal tail 与 prefix 循环重叠 | **不要搬**。缓存要 768 MiB（`README.md:36-38`），本机 4×200K 每卡只剩 326 MB，放不下；算法在 fp16 下与现役内核等价，收益实测约 0 |

一句话：**D256 这个名字指的是注意力头维，那套 workspace 是为"分块重排 P 矩阵"服务的；
真正值钱的是它把 softmax 塞进 GEMM 的 epilogue、以及用 prefix/tail 切分消掉 mask。搬想法，
别搬缓存。**

## 7. 收益总表

基线口径：单条 180K prompt prefill，干净跑 ≈ 83 s（nsys 下 105.37 s / busy 108.45 s，nsys
开销约 21%），4×180K 串行 total ≈ 340 s。本路径单流无重叠（该文件内没有任何
`cudaStreamCreate`/`cudaEventRecord`，全程 `cudaStreamPerThread`），所以 kernel busy 的减少
近似 1:1 反映到墙钟，换算按 83 s 算。

| # | 路线 | 收益（数字） | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|---|
| 1 | **在线单读 softmax** | 内核 −8.2%，省 **1.01 s** busy，**prefill −1.2%**（4×180K ≈ −4 s） | 无条件；fp16/fp8 都适用 | 病态分布多算 `expf`；公共内核需开关兜底 | **未做**，recipe 已实测 + 数值等价已验证 |
| 2 | **不补零**（PV 侧不读被 mask 列） | 内核 −15.0%，省 **1.84 s**，**prefill −2.2%** | 需改 PV 读取语义或把 mask 移到 PV epilogue | 触碰 PV，回归面大 | **未做**，实测数与前提已定 |
| 1+2 | **组合** | 内核 **−20.5%**，省 **2.52 s**，**prefill −3.0%**（4×180K ≈ −10 s） | 以上两条都成立 | 同上 | **未做**，组合已实测 |
| 3a | **softmax 融进 QK GEMM** | 内核 12.278 s → 3~5 s，省 7~9 s，**prefill −8~11%**（单条 −7~9 s） | 自写 sm70 QK GEMM + softmax epilogue | 工程量最大；短上下文可能回退，必须 A/B | **未做**，方向与 1Cat `gemm_with_softmax.h` 一致 |
| 3b | 再融 PV（全融合） | 上限 **prefill −14.8%**（该段内核全免） | 3a 之上再加 PV 融合 | 寄存器/smem 吃紧，易掉 occupancy | 未做，排在 3a 后 |
| 4 | 并掉 update 内核 | 0.641 s busy，≈ **−0.8%** | Step 3 落地后其前提才消失 | 单独做不值得 | 未做，依附 3a |
| C | 向量化 / `exp2f` / warp-per-row / smem 暂存 | −1.5% / −1.1% / **+46%** / −3.5% | — | 都不成立 | **实测否决** |
| D | 搬 1Cat D256 workspace（768 MiB 缓存 + tail stream） | **0**；缓存放不下 | 需每卡 ≥1 GB 空闲 | 显存不够且算法等价 | **否决** |
| E | 软上限：attention 段全清零 | **prefill −14.8%（硬上限）**；含 QK/PV 约 −20%（估算） | 理论上限 | 不现实，用于排优先级 | 参考值 |

**优先级一句话：**attention 桶整体只占 busy 的 12.28%（含 QK/PV 约 20%），**即使全清零，
prefill 也只快 15~20%**。能拿的现实收益是 1.2%（Step 1）到 8~11%（Step 3a）。真正的大头是
GEMM 桶（47.6%，只跑到实测墙的 23~31%），那条路在 `sm70_4x200k_rotate_plan.md` 的 Step 4a/4b。
**本方案建议排在 GEMM 之后，除非 Step 1 顺手就能拿（它无条件、改动小）。**

## 8. 不做什么

- 不做 decode 侧 attention。本方案全程只谈 prefill；decode 那套（Combine 并行化 `debbf431`、
  split/combine、XQA）已有结论，别混进来。
- 不动 KV 量化精度。180K 下 fp4 −8.2%、fp8 −11%（`sm70_status_and_backlog.md:187-198`），
  且 fp8 在 prefill 段解量化只占 0.4%，不是 attention 桶的肉。
- 不为 attention 去动 chunk 尺寸。chunk 8192 的 +7.1% 是既有结论，但 4×200K 下每卡只剩
  326 MB 放不下，本方案不改这个前提。
- 不在没做 A/B 的情况下把 Step 1/2/3 的开关默认打开。
- 不把 busy 占比直接当墙钟收益。所有换算都写明按 83 s 折算，落地上线前必须真跑对照。

## 9. 复跑入口

```bash
# 锁频（结果才可比），跑完记得 -rgc 释放
nvidia-smi -i 0 -lgc 1530,1530
# 真实 trace 几何：rows=4096 ch=8192 base=4096，含正确性对拍与各变体
CUDA_VISIBLE_DEVICES=0 /home/tools/attn_softmax_recommend 4096 8192 40
# 机制判定：去掉 exp、向量化、warp-per-row、smem 暂存
CUDA_VISIBLE_DEVICES=0 /home/tools/attn_softmax_routes 4096 8192 40
nvidia-smi -i 0 -rgc

# trace 归因（device 0）
python3 - <<'EOF'
import sqlite3
d = sqlite3.connect('/home/nsys/pf180k.sqlite')
q = '''select s.value, count(*), sum(k.end-k.start)/1e9
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName
where k.deviceId=0 group by 1 order by 3 desc limit 12'''
print('\n'.join(f'{s:9.3f}s {c:8d}  {n}' for n, c, s in d.execute(q)))
EOF

# 显存带宽屋顶（换算用）
/home/tools/dram_bw
```

**注意：跑这些微基准前先确认 GPU 空闲**（`nvidia-smi --query-compute-apps`）。本方案实测期间
曾有并发的 180K benchmark 占满 4 张卡，导致同一变体量到 0.4202 ms（对 0.2467 ms 偏 +70%），
那次数据已作废。
