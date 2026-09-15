# QPN4 与 QPN8 复看：形状门、实测分布与优先级修正

日期：2026-09-14
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP
输入：`docs/sm70_1cat_port_plan.md`、`docs/sm70_qwen38nvfp4_1cat_gap_audit.md`、
`/tmp/nsys_dec.nsys-rep`（2026-09-14 18:28 抓取）

## 0. 结论

1. **QPN8 不进入本路线。** 它不是「编了没接线」，是缺一整套 FP8 权重契约。本机
   trace 里 QPN8 和 FP8 GEMM 的 kernel 数都是 0，1Cat 自己那条 online QPN8 也
   标着会改变 token 轨迹。
2. **QPN4 不扩覆盖面。** 它的形状门（K%128、N%32、decode 只收 M=1）与 FastLLM
   现有 QPN2 实际生效的门完全相同。GDN-in 两个门都过不了。
3. **现有 plan 的 kernel 占比表是 prefill 污染的结果。** 稳态 decode 单 token 里，
   QPN2 占 41.5% 而不是 1.5%，AllReduce 走的是 FastLLM 自己的 custom kernel
   而不是 NCCL。
4. **GDN-in 每 rank 的 N 是 4120，不是 2608。** FastLLM 已经按 32 对齐把它
   pad 到 4128，和 1Cat 文档里的数字一致。N-pad 这一项比 plan 预期的更小。

## 1. QPN8：不是接线问题

`Fp8QpnGemm` / `Fp8QpnPrepare` 只在三处出现：头文件声明、`qpn8_fp8.cu` 实现、
`sm70QpnFp8Regression` 测试。`src/` 下除 `cuda/sm70/` 目录外零调用。这部分和
audit 的 P1-5 一致。

不一致的是这一项的**性质**。1Cat 的 QPN8 服务的不是 NVFP4 权重，而是
channel-FP8 权重：

| 1Cat 路由表（`docs/design/sm70_qwen38_nvfp4_decode.md`） | TP-local 形状 | 路线 |
|---|---|---|
| FP8 gate/up | K5120 × N8704 | fused QPN8 gate/SiLU/up |
| FP8 down | K4352 × N5120 | QPN8 split-16 |
| FP8 output | K1536 × N5120 | QPN8 split-12 |
| FP8 GDN input、full-attn QKV、LM head | 模型形状 | TurboMind W8A16 |
| NVFP4 gate/up、down | K5120 × N8704 / K4352 × N5120 | TurboMind N32 或 QPN4 |

它的来源是 `sm70_online_qpn8.py`，文件头写得很直白：*requantizes selected dense
checkpoint weights at load time. It is a performance experiment rather than a
precision-preserving checkpoint route*。1Cat 的 2026-08-29 结论也写明，online
QPN8 跑到约 82.27 tok/s 但改变了 token 轨迹并出现一次重复输出，最终验收里
**显式关闭**。

所以 FastLLM 要接 QPN8，先得有一条 channel-FP8（或在线重量化）的权重路线。
这属于改数值契约，不是改分发表。当前 checkpoint 是纯 NVFP4，`lm_head` 被
ignore，接上也无权重可跑。

形状门也比 QPN2 更严：`Fp8QpnCanRun` 要求 `K%16==0`、`N%32==0`、`1<=M<=32`，
block-128 scale 布局还额外要求 `K%128==0 && N%128==0`，且 `M>8` 时
`Fp8QpnGemm` 直接返回 false。

**处置**：保持现状（编译在、单测过、不接线），推迟到有混 FP8 的 checkpoint。
不把它当作「顺手接线」的待办。

## 2. QPN4：与 QPN2 同门，不扩覆盖面

1Cat 的 QPN4 在 `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu`，decode 分支的
`TORCH_CHECK` 是 `k % 128 == 0 && n % 32 == 0`，dispatch 里 `input.size(0) == 1`
之外的 M 全部走 `nvfp4_qpn4_prefill_sm70_out`（解量化到 FP16 dense workspace）。

FastLLM 的 QPN2 门写在 `qpn2_nvfp4.cu:373`，字面是 `K%64==0 && N%32==0`，
但 `Nvfp4QpnCanRun` 末尾还要求 `ChooseSplitK(k) != -1`，而 `ChooseSplitK` 只在
`(k/16) % 8 == 0` 时返回 8，也就是 `k % 128 == 0`。**实际生效的 K 门同样是
128。** N 门和 QPN2 一样是 32。

形状集合相同，所以：

| 投影（TP4 本地） | K | N | QPN2 | QPN4 |
|---|---:|---:|---|---|
| attn QKV | 5120 | 3584 | 过 | 过 |
| attn O | 1536 | 5120 | 过 | 过 |
| MLP gate/up | 5120 | 8704 | 过 | 过 |
| MLP down | 4352 | 5120 | 过 | 过 |
| GDN out | 1536 | 5120 | 过 | 过 |
| **GDN in** | 5120 | **4120** | **不过**（4120%32=24） | **不过**（同门） |

QPN4 相对 FastLLM 现有 QPN2 真正多出来的是三样：融合的 gate/SiLU/up epilogue、
E4M3 scale-code 布局（配 split hi/lo 全局 scale）、以及 NAcc 累加器链与 per-shape
lookahead。这三样都是**同一批形状上的战术**，不是覆盖面的扩张。

第 4 节量的结果是，这三样的上限都不大。

## 3. 实测：一个干净的 decode 单 token

`/tmp/nsys_dec.sqlite` 里混了 prefill 和 decode。以 `lm_head` GEMV
（`FastllmGemvFp16Fp16Kernel2MultiRow`，grid=62080）为界切窗口，可以看到 4 段
prefill（每次 GEMM 的 M=2048，每段约 780 ms kernel 时间、0 个 QPN2、128 个 NCCL）
夹着约 10 个 decode 窗口。其中只有最后一个窗口不含 prefill 工作，是干净的稳态
decode token。

device 0，窗口 t≈2385 ms，1013 个 kernel，合计 12754 µs：

| 组件 | 调用数 | µs/token | 占比 |
|---|---:|---:|---:|
| QPN2 合计 | 208 | 5289.9 | 41.5% |
| ├ N=5120（down / GDN out / attn O） | 128 | 2180.8 | 17.1% |
| ├ N=8704（MLP gate/up） | 64 | 2377.5 | 18.6% |
| └ N=3584（attn QKV） | 16 | 384.4 | 3.0% |
| `FastllmCustomAllReduceKernel<half,4>` | 128 | 2724.7 | 21.4% |
| GDN-in TurboMind `gemm_kernel` | 48 | 1354.5 | 10.6% |
| attention（QGateKV + Split + CombineGQA + SigmoidMul） | 16×4 | 1533.6 | 12.0% |
| `lm_head` FP16 GEMV | 1 | 717.8 | 5.6% |
| RMSNormInner1\<1024\> | 129 | 525.3 | 4.1% |
| RecurrentGatedDeltaRule | 48 | 230.9 | 1.8% |
| AddTo | 128 | 181.8 | 1.4% |
| ShiftAppendConv1DSilu | 48 | 130.8 | 1.0% |
| Swiglu（MLP epilogue） | 64 | 122.7 | 1.0% |
| LinearAttentionStateTranspose | 48 | 114.1 | 0.9% |
| CropNvfp4Output（GDN-in pad crop） | 48 | 67.5 | 0.5% |
| 其余（RMSNormSiluMul、sampling、embedding 等） | | ~200 | 1.6% |

两个交叉验证。总时长 12.75 ms 与之前 env 开关 A/B 的 13.31 ms 基线相差 4%。
QPN2 这 5.29 ms 与 A/B 里「关掉 QPN2 慢 1.58 ms」也自洽，等于它替换掉的那条
路径在这批形状上要花 6.87 ms。

投影普查同样对得上。每个 token 的 NVFP4 投影共 256 条，QPN2 命中 208 条
（128 + 64 + 16），剩下的 48 条正好是 48 个 GDN 层的 `in_proj_qkvzba`，也就是
每 token 那 48 次 TurboMind `gemm_kernel` 加 48 次 crop。

## 4. 带宽账

V100-SXM2 的 HBM 峰值按 900 GB/s 计。

| kernel | 权重字节/token | µs/token | 有效带宽 | 峰值占比 |
|---|---:|---:|---:|---:|
| QPN2 N=5120 | 13.11 MB × 128 = 1678 MB | 2180.8 | 769 GB/s | 85% |
| QPN2 N=8704 | 22.28 MB × 64 = 1426 MB | 2377.5 | 600 GB/s | 67% |
| QPN2 N=3584 | 9.18 MB × 16 = 147 MB | 384.4 | 382 GB/s | 42% |
| GDN-in TurboMind | 10.57 MB × 48 = 507 MB | 1354.5 | 374 GB/s | 42% |
| `lm_head` FP16 GEMV | 635.7 MB | 717.8 | 886 GB/s | 98% |

QPN2 整体 3251 MB / 5289.9 µs = 615 GB/s。这说明三件事。

`lm_head` 已经贴在带宽墙上，没有可动的空间。QPN2 的 N=5120 形状也在 85%，
基本到顶。真正有余量的是 N=8704、N=3584 和 GDN-in 那条 TM 回退，而它们都在
42% 到 67% 之间，属于发射几何和 tile 形状的问题。

按这个账算各项上限（都以 769 GB/s 为目标）：

| 项 | 现在 | 理想 | 上限收益 |
|---|---:|---:|---:|
| GDN-in 改走 QPN2（含 crop） | 1422 µs | ~730 µs | ~0.69 ms，5.4% |
| QPN2 战术把 N=8704 / N=3584 拉到 769 GB/s | 2761.9 µs | 2045 µs | ~0.72 ms，5.6% |
| 融合 gate/SiLU/up epilogue | 122.7 µs | 0 | ≤0.12 ms，1.0% |
| AR 从 21.3 µs/次 降到 1Cat 的 12.4 µs/次 | 2724.7 µs | 1587 µs | ~1.14 ms，8.9% |

注意 QPN4 的收益归属第二行，而它是**战术赌注**：FastLLM 的 QPN2 已经在 615 GB/s
的平均水平上，QPN4 要赢必须赢在发射几何上，这一点本机没有证据。融合 epilogue
那一行只有 1%，远低于 audit 里按 1Cat TurboMind→QPN4 差额估出来的 0.86 ms。

## 5. 对现有 plan 的三处修正

**GDN-in 的 N 是 4120，不是 2608。** 证据是运行时的：pad crop kernel
`CropNvfp4OutputKernel` 的 gridX 是 `(N + 255) / 256`，decode 里 48 次调用的
grid 是 `(17,1,1)`，反推 N 落在 (4096, 4352] 区间。代码侧也对得上，
`BuildQwen35LinearQkvzbaScheme` 在 TP4 下每 rank 取 q(512) + k(512) + v(1536) +
z(1536) + b(12) + a(12) = 4120。audit 里那个 2608 应当改成 4120。

**N-pad 这一项比预期小。** `kPackedOutputAlignment = 32`，所以 TurboMind 回退
路径本来就把 4120 pad 到 4128 再 crop，和 1Cat 文档里「GDN N=4120→4128 padding」
是同一个数。也就是说 pad 已经做完了，缺的只是让 QPN2 也接受 pad 后的 4128 形状
（侧车按 4128 建，输出复用同一个 crop）。kernel 本身不用改，它按运行时 n 取参，
只要求 n%32==0。

**kernel 占比表要重做。** plan 里 NCCL 38%–40%、cutlass FP16 23%–25%、
QPN2 1.4%–1.5% 那一列是全 trace 聚合，prefill 的权重压过了 decode。稳态 decode
里 QPN2 是最大的一族（41.5%），AllReduce 走的是 `FastllmCustomAllReduceKernel`
（128 次/token，21.3 µs/次），NCCL 只出现在 prefill。这不改变 AR 排第一的结论，
但改变了它的依据：要移植的不是「打开 FastLLM 的 custom AR」，因为它**已经在
decode 路径上跑**，要比的是 1Cat 的 pack32 版本能不能把 21.3 µs 压到 12.4 µs。

## 6. 重排后的优先级

按稳态 decode 的实测收益重排：

1. **TP all-reduce**：2724.7 µs/token，占 21.4%，上限约 1.14 ms。这一项的
   前提不变，先做微基准证明 pack32 能赢现有 custom AR 3% 以上。
2. **让 GDN-in 走 QPN2**：1422 µs/token，占 11.1%，上限约 0.69 ms。形状已经
   pad 好，改的是侧车建立与输出 crop，改动比 plan 预期的小。
3. **QPN2 战术（QPN4 一类）**：上限约 0.72 ms，但是赌注，且融合 epilogue 只值
   0.12 ms。放第三，且必须先做算子级 A/B 对上现有 QPN2。
4. **attention / XQA**：1533 µs/token，占 12.0%，但这是短上下文的固定开销。
   长上下文要单独测。
5. **QPN8**：本路线为 0，需要先有 FP8 权重契约。

## 7. 取证方式

```sh
# decode 与 prefill 分层
python3 - <<'PY'
import sqlite3
db = sqlite3.connect('/tmp/nsys_dec.sqlite'); c = db.cursor()
rows = [r[0] for r in c.execute(
    "select k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s"
    " on s.id=k.demangledName where k.deviceId=0"
    " and s.value like '%Fp16Fp16Kernel2MultiRow%' and k.gridX=62080 order by k.start")]
a, b = rows[10], rows[11]
for d, v in c.execute(
        "select k.end-k.start, s.value from CUPTI_ACTIVITY_KIND_KERNEL k"
        " join StringIds s on s.id=k.demangledName"
        " where k.deviceId=0 and k.start>=? and k.start<? order by k.start", (a, b)):
    print(d, v)
PY

# QPN8 在 src 下是否接线
grep -rn 'Fp8Qpn' /home/fastllm/src /home/fastllm/include | grep -v 'cuda/sm70/'

# 本机 trace 里有没有 QPN8 / FP8 kernel
python3 -c "
import sqlite3; c=sqlite3.connect('/tmp/nsys_dec.sqlite').cursor()
print(list(c.execute(\"select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s\"
  \" on s.id=k.demangledName where s.value like '%qpn8%' or s.value like '%fp8%' group by s.value\")))"
```

未验证项：

- 第 3 节的单 token 分布是 n=1。抓取里只有一个不含 prefill 的 decode 窗口，
  其余 9 个都夹了 prefill 分块。总时长与 A/B 基线相差 4%，但严格意义上还没有
  多 token 平均。
- 那次抓取的上下文长度没有从 trace 里确认。attention 的 12.0% 会随上下文变化，
  不能外推到 80K。
- 带宽账按 900 GB/s 峰值算，没有实测本机 HBM 峰值。
- QPN4 的战术收益是上限估计，没有算子级 A/B。要落地必须先量。
