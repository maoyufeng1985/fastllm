# attention 桶优化方案（2026-09-16）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-16GB / TP4 / no-MTP / 180K prompt + FP8 KV。
前提：`sm70_4x200k_rotate_plan.md` 已把 D256 workspace 从 GEMM 路线里划走，指出它的靶子是
attention 桶（12.4%），不是 GEMM 桶（46.7%）。本方案接住那个靶子，回答三件事：attention 桶
里到底是多少、能挤出来多少、1Cat 的 D256 设计能不能搬。

## 0. 一句话结论

**attention 桶的账是真的，但它是个小桶，而且内核已经贴着显存带宽天花板。**实测：把
`FastllmPagedCublasSoftmaxWithCausalMask` 复刻出来，在真实几何下是 0.2486 ms（trace 记录
250.9 µs，复刻对得上），而同样字节数的一次纯拷贝只要 0.1657 ms，纯读只要 0.0782 ms。把
`expf` 换成两个浮点加乘，耗时只降 1.2%，说明瓶颈不是数学，是"同一行反复读"。

能拿的有两条，量级差一个数量级：

- **便宜的一条**：把 softmax 从"读 3 遍"改成"读 1 遍"的在线版，主导几何实测快 13.0%
  （占 91% 的发射），按发射数加权 **−12.8%**，即省掉 1.57 s GPU busy，换算 prefill 墙钟
  **−1.4%（按 busy 108.45 s 算）到 −1.8%（按干净墙 85 s 算）**。数值上现役内核等价（最大
  half 偏差 1e-6，0 个元素超阈，sum 相对误差 3e-7）。
- **有量的一条**：把 softmax 融进 QK GEMM 的 epilogue，让 P 矩阵不落显存。上限就是现在
  浪费掉的那 12.278 s（busy 的 11.3%），现实预期 **prefill −8~−11%**。1Cat 的
  `gemm_with_softmax.h` 干的正是这件事，这也是它那套设计里唯一值得搬的东西。

D256 workspace 本身（768 MiB score 缓存 + tail stream）**不要搬**，理由见 §6：fp16 下它的
算法和现役内核等价，收益实测约 0，而本机 4×200K 场景每卡只剩 326 MB，它的缓存直接放不下。

## 1. 两个桶的账，先把归属钉死

180K prefill 实测（`/home/nsys/pf180k.sqlite`，180K prompt + chunk 4096 + 200K 池 + FP8 KV +
low_gpu_mem，device 0，窗口 118.20 s、kernel busy 108.45 s = 91.8%）：

| 内核族 | 每卡时长 | 占 busy | 说明 |
|---|---:|---:|---|
| cutlass Kernel2（NVFP4 GEMM） | 39.160 s | 36.1% | 253696 次发射 |
| NCCL AllReduce RING_LL | 36.099 s | 33.3% | PCIe 结构性成本 |
| **`FastllmPagedCublasSoftmaxWithCausalMask`** | **12.278 s** | **11.3%**（占整个窗口 10.4%）| **attention 桶的主体，49056 次发射** |
| volta_s884gemm_fp16（FP16 GEMM 三族之一，最大） | 9.076 s | 8.4% | QK/PV 的 cuBLAS 就算在这里 |
| volta_h884gemm_64x64 | 2.931 s | 2.7% | |
| `FastllmCudaNVFP4Block162HalfKernel`（反量化） | 2.070 s | 1.9% | |
| `FastllmPagedCublasAttnBlockUpdateFloat` | 0.641 s | 0.6% | 在线 softmax 的 running max/sum 合并 |
| `FastllmPagedCacheGatherHeadRangeKernel`（FP8 KV 解量化搬移） | 0.401 s | 0.4% | |
| `FastllmPagedAttentionSplitTiledGqaKernel` | 0.050 s | 0.05% | 是 decode 侧残留，prefill 不用 |

两点记账修正，和旧文档的 12.4% 对得上但不完整：

1. **attention 桶 13.320 s = softmax 12.278 + update 0.641 + gather 0.401，占 busy 的
   12.28%**（对得上旧口径的 12.4%），其中 softmax 一个内核占桶的 92%。
2. **QK/PV 的 GEMM 不在这个桶里**。prefill 的 QK/PV 走 `cublasHgemm`，在 trace 里落进
   "FP16 GEMM 三族 12.6 s（11.6%）"那个桶，无法从 trace 里把 attention 的部分单独拆出来。
   所以 attention 侧真正共享的算力比 12.28% 高一些（含 QK/PV 约 20%，属估算）。做内核优化
   按 20% 估天花板，算 Step 1/2 的直接受益按 12.28% 的 softmax 桶算，两个数不要混。
3. **GEMM 口径的对照**：Kernel2 36.1% + FP16 GEMM 三族约 11.6% = 47.6%（本方案复算）。
   旧文档写 46.7%，差在 FP16 GEMM 三族的取法不同，两者都可接受，不以此为准。

模型几何（`/home/models/Qwen3.8-27B-QUASAR-NVFP4/config.json`）：64 层里只有 16 层是
`full_attention`，另外 48 层是 `linear_attention`；`head_dim=256`、`num_attention_heads=24`、
`num_key_value_heads=4`，TP4 下每卡 1 个 KV head、6 个 Q head（`group=6`）。这解释了为什么
attention 只占 12.4%：只有 1/4 的层在做 O(context²) 的活。

## 2. 归因实测：从 trace 反推出真实几何

同一个内核在 trace 里的形状（device 0）：

| gridX | 发射次数 | 总时长 | 单次 |
|---:|---:|---:|---:|
| 4096 | 46848 | 11.755 s | 250.9 µs |
| 3872 | 2112 | 0.523 s | 247.6 µs |
| 16 | 96 | 0.000 s | 3.5 µs |

关键在这一列：**46848 次发射的耗时全都一样（直方图 260 us 档占 38906 次），而因果三角的
工作量本该随行号增长。**耗时恒定只有一个解释：每一行的 `visible == channels`，即 KV 窗口
完全可见。这正是深上下文（180K）的特征：`base = kvLen - qoLen - kvStart` 在累积到 4096 列
之后就不再小于 0，`fastllm-paged-attention-native.cu:412-414` 那个补零循环因此是空转。

由此反推出调用参数：`rows = qoLen = 4096`、`channels = chunkLen = 8192`，且约 91% 的发射
（46848/49056）走的是"整行可见"这条几何。这一条决定了后面所有测量的形状，也顺手否定了一个
看着很香的路线：补零那趟不是浪费（§5 路线 C）。

复算一遍流量：每次发射动 `4096 × 8192 × 2 B = 67.1 MB`，46848 次 = 3.14 TB 的 P 矩阵；
softmax 读 3 遍写 1 遍 = 12.6 TB；按 12.278 s 算有效 1.03 TB/s，已经**高于**本机实测的
拷贝屋顶 791 GB/s 和读屋顶 893 GB/s（`/home/tools/dram_bw`），多出来的部分是 L2 命中。
换句话说：这个内核没有留下"浪费掉的带宽"，它只是在把同一份数据搬太多遍。

## 3. 机制判定：复刻内核 + 变体实测

复刻在 `/home/tools/attn_softmax_routes.cu`、`attn_softmax_bench.cu`、`attn_softmax_online.cu`，
V100 锁频 1530 MHz（`nvidia-smi -lgc 1530,1530`），`rows=4096 ch=8192 reps=40`，可控可复跑。

对照点：复刻版 0.2486 ms vs trace 250.9 µs，复刻是可信的。

| 变体 | 耗时 | 相对现役 | 逻辑流量 | 结论 |
|---|---:|---:|---:|---|
| 现役结构（3 读 + 1 写 + 补零） | 0.2486 ms | — | — | 基线 |
| 同样结构，`expf` 换成 2 个 FLOP | 0.2456 ms | −1.2% | — | **数学不是瓶颈** |
| `exp2f` 替代 `expf` | 0.2458 ms | −1.1% | — | 同上，这条路口已关 |
| half2 / half4 向量化 | 0.2449 ms | −1.5% | — | 访存宽度不是瓶颈 |
| warp-per-row（去掉块级归约） | 0.3643 ms | +46% | — | 变慢，块级归约不是瓶颈 |
| 共享内存暂存（DRAM 只读一遍） | 0.2398 ms | −3.5% | 省一半读 | **不划算**：smem 挤掉 occupancy |
| 在线单读（2 读 + 1 写） | 0.2299 ms | −8.2% | 少一遍读 | 有效 |
| 在线单读 + `exp2f` | 0.2252 ms | −10.1% | 少一遍读 | 有效 |

"整行可见"几何下（占 91% 的发射）：

| 变体 | 耗时 | 相对现役 |
|---|---:|---:|
| 现役结构 | 0.3006 ms | — |
| 在线单读 | 0.2617 ms | **−13.0%** |
| 在线单读 + `exp2f` | 0.2517 ms | **−16.3%** |
| 纯拷贝屋顶（1 读 + 1 写） | 0.1657 ms | −44.9% |
| 纯读屋顶（1 读） | 0.0782 ms | — |

数值等价性（同一输入，与现役内核逐元素比）：

| 变体 | 最大 half 偏差 | 超 1e-3 的元素数 | sum 相对误差 |
|---|---:|---:|---:|
| 在线单读 | 0.000001 | 0 | 2.84e-07 |
| 在线单读 + exp2 | 0.000001 | 0 | 3.18e-07 |

这两行是整个方案的关键：**在线版把一趟读省掉，数值上和现役内核等价。**

## 4. 为什么以前说"没有白捡的内核优化"，这次不一样

不矛盾，是两次量的不是同一个东西。

旧结论（`sm70_1cat_port_plan.md:284-301`）测的是 **decode** 侧同几何探针：读天花板 786 GB/s
= 88% 屋顶，加上真实在线 softmax 后掉到 456 GB/s = 51%，于是判定"瓶颈是 softmax 数学吞吐"。
那个判定对 decode 成立（S=192，行很短，归约开销盖过访存）。

本次测的是 **prefill** 侧的实际内核：`expf` 换掉只降 1.2%、`exp2f` 不降、向量化不降、把
warp 归约去掉反而变慢，四路交叉都指向"不是数学"。而且 prefill 的桶不是 11.8%，是 12.4%
且几何完全不同（行宽 8192 且整行可见）。**结论：prefill 的 softmax 是"趟数瓶颈"，decode 的
才是"数学瓶颈"，两者不能互相引用。**

## 5. 分步方案

### Step 1. 在线单读 softmax（便宜，先拿）

- 改什么：`FastllmPagedCublasSoftmaxCausalFunc`（`fastllm-paged-attention-native.cu:355-415`）
  现在是 4 趟（367 求 max / 387 求 sum / 410 归一化写出 / 412 补零），改成 2 趟：第一趟
  每线程跑在线 (m, s) 并在块级用带 rescale 的归约合并，第二趟读第二遍、归一化、写出，补零
  那趟保留（整行可见时它是空循环，不进主循环代价）。
- 开关：新增 `FASTLLM_PAGED_SOFTMAX_ONLINE`，默认关，A/B 用；现役结构保持可回归。
- 验证门：`FASTLLM_PAGED_SOFTMAX_ONLINE=1` 与默认在同一 180K 配置下跑，要求
  (a) 输出 sha256 与默认一致或 `bits_diff` 为 0，(b) prefill tok/s 提升 ≥3%，
  (c) 4×200K 场景 exit 0 无 OOM。
- 风险：在线版在"元素大于 running max"时才多算一次 `expf`；本次用随机分布实测没退化，
  但要补一个极端分布（单行内量级跨 20）的用例，确认没有病态退化。

### Step 2. softmax 融进 QK GEMM epilogue（有量，主菜）

- 为什么这是主菜：现在 P 矩阵要走"QK 写一遍 → softmax 读 3 遍写 1 遍 → PV GEMM 再读一遍"，
  合计 12.6 TB 的显存流量换一个中间结果。把这个中间结果留在寄存器/smem 里，softmax 的
  12.278 s 就基本消失。上限就是它（busy 的 11.3%），现实预期 −8~−11% prefill。
- 怎么落地：现在是 `cublasHgemm` 直调（`:944`、`:1027`），cuBLAS 不给 epilogue 挂钩，所以
  要自写一个 sm70 的 QK GEMM 带 softmax epilogue。这正是 1Cat
  `csrc/attention/sm70_v37/gemm_with_softmax.h` 的做法（M128/N256/K32 tile + softmax
  visitor + 同流上的一次 final-reduction），也是 §6 里 D256 那套设计真正值钱的部分。
- 分两步走，别一次吞完：
  - 2a：只融 softmax（QK GEMM epilogue 算 exp、写归一化后的 P），P 仍写一次给 PV 读。
    预期 softmax 的 12.278 s 降到约 3 s（就剩写 P），prefill −8%。
  - 2b：再融 PV（P 完全不落 DRAM）。上限 −16%，但寄存器/smem 会吃紧，先不做。
- 验证门：与 fp16/fp8 两条 KV 路径各跑一次 180K A/B，要求输出一致（`bits_diff=0`），
  prefill tok/s ≥ +6%，且 8K/80K 短上下文不回退（短上下文下融合的收益变小，别做成倒退）。
- 风险：自写 sm70 GEMM 是本方案最大工程量，cutlass 在 7.0 上要避开 tensor-core 的
  tf32/bf16 假设；收益取决于 P 是否还得 round-trip 一次（2a 拿不满）。

### Step 3.（可选）把 update 内核并掉

`FastllmPagedCublasAttnBlockUpdateFloat` 只有 0.641 s（0.6%），但如果 Step 2 的 epilogue 里
已经有 running max/sum，这个内核的存在前提就没了，跟着删。单独做不值得。

## 6. 1Cat 的 D256 设计，哪些能搬、哪些不能

读的是 `/home/1Cat-vLLM/csrc/attention/sm70_v37/`（`prefill.cu` 666 行、`tail.cu`、
`gemm_with_softmax.h`、`README.md`）。它由三块组成，逐块判断：

| 组成 | 它做什么 | 能不能搬 |
|---|---|---|
| QK GEMM 带 softmax epilogue（`gemm_with_softmax.h`） | QK 算完直接算 softmax，P 不落显存 | **能，且值得**。这就是 Step 2；是这套设计里唯一有量的一块 |
| prefix/tail 因果切分 | `prefix = total_kv - kTail`（`prefill.cu:481-487`）后，prefix 无 mask、tail 走标准 FA2 | 只对本机 9% 的发射有用（整行可见时没有 mask 可省）。可作为 Step 2 的附赠优化 |
| 768 MiB score 缓存 + tail stream（`prefill.cu:382-418`、`:560-661`） | 缓存 49152×8192 的 fp16 score 张量，用私有流把 causal tail 与 prefix 循环重叠 | **不要搬**。缓存要 768 MiB（`README.md:36-38`），本机 4×200K 每卡只剩 326 MB，放不下；它的算法在 fp16 下和现役内核等价，收益实测约 0 |

一句话：**D256 这个名字指的是注意力头维，那套 workspace 是为"要分块重排 P 矩阵"服务的；
真正值钱的是它把 softmax 塞进了 GEMM 的 epilogue。搬想法，别搬缓存。**

## 7. 收益总表

基线：单条 180K prompt prefill（chunk 4096、FP8 KV、干净跑约 85 s，kernel busy 108.45 s 的
测量来自 nsys 带 21% 开销的那次；换算按 busy 占比折算到干净墙上）。4×180K 串行 total ≈ 340 s。

| 路线 | 收益（数字） | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|
| **Step 1 在线单读 softmax** | softmax −12.8%（发射加权），省 1.57 s busy，**prefill −1.4%**（单条 ≈ −1.2 s；4×180K ≈ −5 s） | 单流；fp16/fp8 都适用；保留 running max/sum 语义 | 病态分布下多算 `expf`；动的是所有 prefill 请求都走的公共内核 | **未做**，recipe 已实测并验证数值等价 |
| **Step 2a softmax 融进 QK GEMM** | softmax 12.278 s → 约 3 s（省约 9.3 s busy），**prefill −8%**（单条 ≈ −7 s；4×180K ≈ −27 s） | 需自写 sm70 QK GEMM + softmax epilogue；P 仍写一次给 PV 读 | 工程量最大；短上下文可能回退，必须做 A/B | **未做**，方向与 1Cat `gemm_with_softmax.h` 一致 |
| Step 2b 再融 PV（全融合） | 上限 **prefill −16%**（P 完全不落 DRAM） | 以上再加 PV 融合 | 寄存器/smem 吃紧，容易掉 occupancy | 未做，排在 2a 之后 |
| Step 3 并掉 update 内核 | 0.6% busy，≤0.5% prefill | Step 2a 落地后其前提才消失 | 单独做不值得 | 未做，依附 2a |
| 路线 C：省掉补零那趟 | **0**（整行可见时是空循环，占 91% 的发射） | — | 只对 9% 的因果发射有效 | **否决**，实测无收益 |
| 路线 D：搬 1Cat D256 workspace | **0**（fp16 下算法等价）；缓存要 768 MiB 放不下 | 需每卡 ≥1 GB 空闲显存 | 显存直接不够，收益为零 | **否决** |
| 路线 E：fp8/fp4 KV 走 D256 专用内核 | 未知，但方向与已测结论冲突（180K fp4 −8.2%；fp8 在 prefill 段解量化只占 0.4%） | — | 大概率负收益 | **否决** |
| 路线 F：向量化 / `exp2f` / warp-per-row / 共享内存暂存 | −1.5% / −1.1% / +46% / −3.5% | — | 都不成立 | **实测否决**，四个方向各测一次 |

**天花板提醒：** attention 桶整体只占 busy 的 12.4%（把 QK/PV GEMM 算进来约 20%），所以
**即使把 attention 全部清零，prefill 也只快 12~20%**；上面能拿的现实收益是 1.5%（Step 1）
到 8~11%（Step 2）。真正的大头是 GEMM 桶（46.7%，只跑到实测墙的 23~31%），那条路在
`sm70_4x200k_rotate_plan.md` 的 Step 4a/4b。**本方案排在 GEMM 之后做，除非 Step 1 顺手就能拿。**

## 8. 不做什么

- 不做 decode 侧的 attention 优化。本方案全程只谈 prefill，decode 那套（Combine 并行化
  `debbf431`、split/combine、XQA）已有结论，别混进来。
- 不动 KV 量化精度。180K 下 fp4 −8.2%、fp8 −11%（`sm70_status_and_backlog.md:187-198`），
  且 fp8 在 prefill 段的解量化只占 0.4%，不是 attention 桶的肉。
- 不为了 attention 去动 chunk 尺寸。chunk 8192 是 Step 2 的既有结论（prefill +7.1%），
  但 4×200K 下每卡只剩 326 MB 放不下，本方案不改这个前提。
- 不在没做 A/B 的情况下把 Step 1/2 的开关默认打开。

## 9. 复跑入口

```bash
# 锁频（结果才可比），跑完记得 -rgc 释放
nvidia-smi -i 0 -lgc 1530,1530
CUDA_VISIBLE_DEVICES=0 /home/tools/attn_softmax_online 4096 8192 40 8191   # 整行可见几何
CUDA_VISIBLE_DEVICES=0 /home/tools/attn_softmax_online 4096 8192 40 4096   # 因果三角几何
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
```
