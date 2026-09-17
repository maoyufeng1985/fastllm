# FlashInfer pcie_ipc all-reduce 接入 FastLLM：实现记录

日期：2026-09-17
对象：4×V100-SXM2-16GB / SM70 / PCIe gen3 x16 / 四卡同在一个 PIX switch / TP4
上游：FlashInfer `comm/pcie_ipc_all_reduce.cuh`（取自上游 `main`，已收进
`third_party/flashinfer/comm/`）

---

## 0. 一句话

**把 FlashInfer 的 PCIe one-shot all-reduce 接进引擎，只在 160 KiB 以下替掉 NCCL，
默认关闭。** 微基准显示这一档它快 1.3–1.9 倍且位级正确；**引擎里的收益还没有数**，
见 §5。

---

## 1. 为什么它会快，以及为什么只在 160 KiB 以下

`pcie_ipc` 的定位（引自它的头部注释）：**没有 NVLink、每张卡都经 CPU root complex
互访的机器上，全对全写会塌成单点带宽的一小部分，所以这些 kernel 把推送分阶段，
使任一时刻每个 rank 恰好只有一条出站和一条入站流。** 它治的是"小消息的固定开销 +
交叉写"，不是大消息的带宽。

**本机实测（同一进程、同一循环、同尺寸、同计时口径，FP16，eager，各 60 次）**：

| bytes | FlashInfer µs | NCCL µs | FI/NCCL | 正确性 |
|---:|---:|---:|---:|---|
| 4 KiB | 21.37 | 40.98 | **0.521×** | 两边都 PASS（位级一致） |
| 40 KiB | 30.53 | 48.26 | **0.633×** | 同上 |
| 160 KiB | 61.03 | 78.42 | **0.778×** | 同上 |
| 512 KiB | 153.80 | 116.92 | 1.315× | 同上 |
| 1 MiB | 311.33 | 191.23 | 1.628× | 同上 |
| 4 MiB | 1347.16 | 590.85 | 2.280× | 同上 |
| 8 MiB | 2668.58 | 1135.02 | 2.351× | 同上 |

正确性判据是位级可判且与归约顺序无关的输入：第 r 张卡全填 `2^-r`，四卡和
= 1.875，全是 2 的幂，任何求和顺序在 fp16 下结果相同。两边都得到 `0x3f80`、
逐元素 0 个不符。

**交叉点在 160 KiB 与 512 KiB 之间。** 所以默认上限取 `160 KiB`
（`FASTLLM_PCIE_IPC_AR_MAX_BYTES` 可改），超过就退回 NCCL。

## 2. 实现

| 文件 | 作用 |
|---|---|
| `include/devices/multicuda/pcie_ipc_ar.h` | 接口：`Init` / `Ready` / `MaxBytes` / `TryAllReduce` / `Shutdown` / `Uses` |
| `src/devices/multicuda/fastllm-pcie-ipc-ar.cu` | 实现 |
| `src/devices/multicuda/fastllm-multicuda.cu` | 在 `FastllmInitNccl` 里 `Init`，在 `FastllmNcclAllReduceImpl` 里分派 |
| `CMakeLists.txt:480` | 把新 `.cu` 加进 `FASTLLM_CUDA_SOURCES` |
| `third_party/flashinfer/comm/pcie_ipc_all_reduce.cuh` | 上游头文件，自包含（只依赖 `cuda_bf16.h`/`cuda_fp16.h`/`cuda_runtime.h` + 三个标准头） |

**接线要点**

1. **不用 CUDA IPC。** 上游假设"每卡一进程"，用 CUDA IPC 共享 workspace；FastLLM 是
   单进程多卡，各 rank 的 `cudaMalloc` 指针全进程可寻址，所以直接把它们交给
   `make_peer_views` 即可。**本机 `cudaIpcOpenMemHandle` 一律失败**
   （跨进程 `invalid resource handle`，同进程自开自己导出的句柄
   `invalid device context`），所以这一步不是优化，是必需。
2. **分派条件**：`defaultStream && allowCustomAllReduce`。
   - `defaultStream` 是必需的：胶水层在 `cudaStreamPerThread` 上发射，侧流调用若进来
     会落到与调用方排序的不是同一条流。
   - `allowCustomAllReduce` 沿用引擎既有的语义（侧流路径传 false）。
3. **排空**：成功发射后**必须**照 NCCL 路径一样做
   `cudaStreamSynchronize(stream)`。SM70 上每个集合通信后的排空是 load-bearing 的，
   注释在 `FastllmNcclAllReduceImpl` 里写明它防的是"在途集合通信 vs 真实
   cudaMalloc 抢 CUDA 驱动锁导致的跨 rank 死锁"。**绕过它会造成典型悬挂**
   （一卡 0%、其余三卡 100%）。
4. **in-place 用独立暂存**。kernel 一边读输入一边写输出，`data == dest` 时要先拷到
   暂存。**暂存不能借用 workspace 本体**——workspace 头几个字节就是 kernel 用来
   同步的 signal 槽，写激活值会破坏协议。所以另分配一块 `MaxBytes` 大小的暂存。
5. **`Init` 不做任何 CUDA 调用**。它在 `FastllmInitNccl` 里被调用，而那里持有
   `g_ncclInitMutex`，其余 rank 线程全部排在锁上。此时从本线程去碰别的设备的上下文
   会等一个无法推进的线程。改为**各 rank 在自己的线程上懒分配**（`EnsureLocal`），
   `Init` 只记账；等待 peer 发布带 10 秒上限，超时即失败退回 NCCL。
6. **`Init` 幂等**。`FastllmInitNccl` 的 `ready` 分支会重复走初始化路径，重复 `Init`
   对同一设备组是 no-op，否则会把已发布的 slab/计数清零，造成 rank 间路径分歧。

## 3. 接入过程中修掉的四个 bug（留痕）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `no CUDA-capable device is detected`（启动即失败） | 不重要：是设备当时已掉，见 §6 | — |
| 2 | 120 秒 TIMEOUT，日志停在 `pcie_ipc AR armed` 之前 | `Init` 持有 NCCL 初始化锁去做四卡的 `cudaSetDevice`/`cudaMalloc`，其余 rank 线程全排在锁上 | 改为每 rank 自己线程懒分配，`Init` 不发 CUDA 调用 |
| 3 | 40 秒 HANG，现场"GPU0 0% / 其余 100%"、无线程在 CUDA 调用里 | 成功后直接 `return`，跳过了 SM70 必需的每集合通信排空 | 补上与 NCCL 路径相同的排空 |
| 4 | in-place 会踩 workspace 的 signal 槽 | 暂存借用了 workspace 本体 | 另分配独立暂存缓冲 |

| 5 | 每次 NCCL 重建泄漏一份 slab + 一份暂存 | `Shutdown` 有意不释放 slab，却在注释里把它当"故意泄漏"，而 `Init` 下次会把指针清成 0，旧分配就再也找不回来 | `Init` 开头用 `exchange(0)` 回收上一轮的 slab/暂存再武装 |
| 6 | **rank 之间的超时不对称，会造成路径分歧死锁** | `EnsureLocal` 等 peer 发布时，每个 rank 各用自己的起始时刻算 10 秒超时。一个 rank 超时退回 NCCL，而另一个 rank 稍晚在它自己的截止前等到全部发布并发射了 pcie_ipc——两边进入不同集合通信，永久互等 | 超时改成**共享的绝对截止时刻**（`Init` 时设定），所有 rank 要么一起成功、要么一起失败；并显式检查共享的 `failed` 标志 |

另外修了一个**测试工具**的缺陷：逐 rank 追踪原先用三次 `fprintf` 写 stderr，
四个 rank 线程会交错，日志里出现过 `dev=3 DECLINE ...dev=0 DECLINE ...` 粘在一行。
这导致我一次误判"rank 0 少走一次 = 路径分歧"。改成单次 `fwrite` + 互斥后不再交错。

**第 6 条是这次实现里最值得记住的一条**：它和前两次 HANG 的形状完全一样
（一卡 0%、其余 100%），但根因不同。#3 是漏了排空、#6 是超时不对称，
两者都会表现为"某个 rank 不在集合通信里"。**排查这类问题时，"谁没进场"比
"谁卡住了"更有信息量。**

## 4. 怎么跑

```sh
# 关（基线）
FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 \
  python3 -m ftllm.cli benchmark <模型> --tp 4 --input_tokens 8192 --output_tokens 8 ...

# 开
FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 FASTLLM_PCIE_IPC_AR=1 \
  python3 -m ftllm.cli benchmark <模型> --tp 4 --input_tokens 8192 --output_tokens 8 ...

# 附加：调整上限 / 打开逐 rank 追踪
FASTLLM_PCIE_IPC_AR_MAX_BYTES=524288      # 上限（默认 163840）
FASTLLM_PCIE_IPC_AR_TRACE=1               # 每个 rank 的 TRY/ENTER/READY/LAUNCHED/DECLINE
```

判据（跑之前定死）：
- **P1 可达**：开启态日志必须出现 `pcie_ipc AR armed`，且四个 rank 的 `LAUNCHED`
  次数一致。
- **P2 数值**：开启态与关闭态的 token `sha256` 必须完全一致，否则作废。
- **P3 收益**：用**同一次开机、同一二进制、同一环境**的组内极差做判据
  （实测 off 三样本极差 **0.0012 s**，见 §5.2；不要再用早先那个跨编译的 0.0417 s）。
  但更硬的前提是**先排除臂序混淆**：见 §5.1，三次开机都是 off 先。

## 5. 收益表

| 路线 | 数字 | 成立条件 | 主要风险 | 状态 |
|---|---|---|---|---|
| 微基准：≤160 KiB 替 NCCL | **快 1.29–1.92 倍**（本机实测，见 §1） | 单进程多卡 + UVA，FP16 | 无（位级正确） | **已测** |
| 引擎：小消息档换 pcie_ipc | **off 3 样本 mean=3.2496（3.2491/3.2494/3.2503），on 3 样本 mean=3.2377（3.2410/3.2321/3.2399）；均值差 0.0119 s = 0.37%，两组不重叠（分离 0.0081 s），三个独立开机都同向**；两臂 sha 全为 `adcb1bd7` | 需要"on 先跑"的一次开机才能排除顺序混淆 | **顺序被混淆：三次都是 off 先、on 后**，第二臂更快也可能是热身 | **已测，方向一致但归属未确证** |
| 引擎：>160 KiB 也换 | **未做、无数据**，且微基准显示 ≥512 KiB 时 NCCL 快 1.3–2.35 倍 | — | 会把大消息做慢 | **不做**（上限即防此） |

### 5.1 三组 A/B 样本与顺序混淆（必须说清）

| 臂 | 样本（Total, s） | 均值 | 组内极差 | sha | served |
|---|---|---|---|---|---|
| off | 3.2491 / 3.2494 / 3.2503 | **3.2496** | 0.0012 | 全是 `adcb1bd7` | 0 |
| on  | 3.2410 / 3.2321 / 3.2399 | **3.2377** | 0.0089 | 全是 `adcb1bd7` | 8192（后两次有计数，第一次没有） |

两组**不重叠**（`off` 最小 3.2491 > `on` 最大 3.2410，分离 0.0081 s），且**三个独立开机
都同向**。来源：`/tmp/AB_off.out`、`/tmp/R_off1.out`、`/tmp/L_off3.out`、
`/tmp/AB_on.out`、`/tmp/R_on1.out`、`/tmp/L_on3.out`。

**缺陷（必须标出来，否则这个数字会误导）**：三次开机的**臂序都是 off 先、on 后**，
从没跑过 on 先。所以"on 更快"与"第二臂更快"在这批数据里**完全混淆**——
第二臂更快也可能只是热身效应（页面缓存、模块加载残留），与 pcie_ipc 无关。
**因此这 0.37% 不能归因给 pcie_ipc。** 要归因，需要一次"on 先跑"的开机。

### 5.2 更正：先前引用的 0.0417 s 不是同类离散

我在早先几轮里用"本机 8K 离散 0.0417 s"当判据，那是**跨多次重新编译**的 8 次读数极差
（`3.2151` 到 `3.2568`，中间换过好几次二进制，还混了 `NCCL_DEBUG` / 统计开关等不同
环境），**不是同一二进制、同一环境下的 run-to-run 离散**。同类离散要小得多：

- 同一次开机内、同一二进制、同一环境的 off 三样本：极差 **0.0012 s**。
- 跨三次开机（三次重启）的 off 三样本：极差仍是 **0.0012 s**（3.2491/3.2494/3.2503）。

**所以判据口径要改**：不能再用 0.0417 s。但改了判据也不改变上面的结论——
**顺序混淆没有排除之前，差值再小也不能归因。**

### 5.3 反转臂序实验：预测与判据（跑之前写死）

**设计**：下一次开机只跑 2 臂，顺序反过来——**`on` 第一臂、`off` 第二臂**。

**为什么这个设计能分辨**：两臂的"冷/热"位置互换了。

| 若真实原因是… | 反转后应看到 | 与已有数据的关系 |
|---|---|---|
| **pcie_ipc 真的更快** | `on`（冷）仍比 `off`（热）快 | 与三对旧样本同向 |
| **只是第二臂更快（热身）** | `on`（冷）比 `off`（热）**慢** | 与旧样本反向，差值符号翻转 |

**预测（写在这里，事后对照）**：
- 若 pcie_ipc 有效：`onFirst ≈ 3.23x`，`offLast ≈ 3.24x`
- 若纯热身效应：`onFirst ≈ 3.24x–3.25x`，`offLast ≈ 3.23x`

**判据**：看差值的**符号**，不看大小。符号与旧样本一致 → 归因给 pcie_ipc
（且与热身无关，因为冷的那个更快）；符号翻转 → 是热身效应，0.37% 作废。
**两种结果都要写进本节，不许只留支持后者的那份。**

### 5.4 "一次开机 2 臂"这条规则的依据强度（诚实标注）

规则来自 2 次观测：两次开机的**第 3 臂**都掉卡。但**臂序与"开机后经过的时间"
在这 2 次里是共线的**，无法区分：
- 会话 A：开机 → 第 1 臂(10:57:03) → 第 2 臂(10:57:37) → 第 3 臂(10:59:06，崩于 11:00:40)
- 会话 B：开机(11:01:30) → 第 1 臂(11:03:53) → 第 2 臂(11:04:27) → 第 3 臂(11:04:58，崩于 11:04:58)

两次崩溃都在**开机后 200–300 秒**、且都是第 3 臂。**"第 3 臂"和"开机后 ~4 分钟"
是同一件事的两种说法**，本会话没有把两者分开的实验。

**所以规则的表述按"2 臂"来写（可操作），但它的机制标为未确定**：
可能是累积 GPU 工作、可能是时间/热，也可能与臂数本身无关。

**引擎这一次的可达性已单独验证**（引自 `/tmp/REACH.out`，`FASTLLM_PCIE_IPC_AR_TRACE=1`）：
四个 rank 各 `TRY=128 / LAUNCHED=128 / DECLINE=51`，**次数完全对称**，
51 次被拒全部是 40 MiB 的 prefill AR 撞 160 KiB 上限（设计行为）。
**没有这份计数时，`armed=1` 只能证明 `Init` 跑过，不能证明 kernel 发射过**——
第一次 A/B 就是这个问题，两次读数可能构造上相同。

**为什么引擎收益不能拿微基准的倍数去乘**：微基准是独立循环、独立缓冲、固定尺寸；
引擎里是真实前向、张量与 GEMM 交错、流拓扑不同，而且我改了通信建立方式
（UVA 而非 IPC）。两者是**不同对象**，只能各自测。

## 5.5 复查中又修掉两个 bug（2026-09-17 晚）

按`AGENTS.md`的"假 PASS"教训复查自己的代码时找到的。两处都不是性能问题，是正确性问题。

| # | bug | 为什么会坏 | 修法 |
|---|---|---|---|
| 7 | **`s.views` 用 `std::vector`，读它时没加锁** | `Init`/`Shutdown` 在锁内 `assign`/`clear`，而 `TryAllReduce` 在**锁外**读 `s.views[rank]`。引擎重新分组时会再次进 `FastllmInitNccl`，vector 可能重新分配，于是某个 rank 拿着**已释放的存储**去发射 kernel（非法访存） | 换成定长数组 `fi::PeerViews views[kMaxWorld]`，彻底消除重新分配；另加"代数"计数，`Init` 每次重新武装就 +1，发射前校验视图属于当前代数，不属于就拒绝 |
| 8 | **120 秒等待上限从 `Init` 时刻起算** | `Init` 在 NCCL 初始化时执行，而**模型加载 + 预热常常超过 120 秒**；等第一次真的用到这个功能时，上限早已过期，四个 rank 一起失败、静默退回 NCCL，**功能被无声关掉** | 上限改成**首次使用时才武装**（第一个到达的 rank 设，四个 rank 用同一个绝对时刻），并且 `Init` 时重置 |

**顺带留一条不成立的推断（撤错留痕）**：我曾推断"`FASTLLM_PCIE_IPC_AR_MAX_BYTES` 设成非 8 的倍数会让 `maxNumel` 不整除、包索引错位、数值出错"。核实后**不成立**：`rank_stride_packs = max_numel / kPackElems`（`pcie_ipc_all_reduce.cuh:2247-2248`）只是每一步的均匀步长，截断不会越界——每块的写入量是 `numel`，而 `numel <= max_numel`。**所以这条不改，也不当 bug 报。**

### 5.6 这两个 bug 怎么才能被测出来（说清手段）

**bug 7 和 8 都只能在"引擎重新分组"或"加载超过 120 秒"的路径上触发**，而本会话的普通 8K 跑**走不到这两条路**——所以它们不代表已测出的性能差异失真，
而是**这两条路从未被测过**。

- bug 8 的判据（便宜）：一次 8K 跑，开启态打印的
  `pcie_ipc AR: <n> collective(s) served` 是否 > 0。**本次开机已经测过：`served=8192`**，
  说明 120 秒上限**在这次跑里没有过期**（模型加载较快）。要触发它需要更慢的加载路径。
- bug 7 的判据（不便宜）：需要让引擎**重新分组**（`FastllmInitNccl` 重入）之后仍发
  pcie_ipc 调用。本会话没有构造过这个场景，**标记为未验证**。

## 6. 本次会话的硬件事故（与代码无关，但影响所有后续计划）

**会话内掉卡三次**（2026-09-17 10:26 补记第三次，发生在本 A/B 的"开启态"那一臂运行期间），
三次都是同一个 `pcieport 0000:04:04.0`（PLX switch）报
`Multiple Uncorrectable (Non-Fatal) error`，随后四卡
`nvidia 0000:05/07/08/09:00.0: AER: can't recover (no error_detected callback)`、
`Xid 79`（GPU has fallen off the bus）+ `Xid 154`（GPU Reset Required），
`AER: device recovery failed`。恢复只能靠重启机器。

**第三次的时间线**（观测）：Xid 79/154 全在 uptime 1799 秒；`pcieport 0000:04:04.0`
报 `Multiple Uncorrectable (Non-Fatal)`，四卡 `AER: can't recover` → `Xid 79`
（fallen off the bus）+ `Xid 154`（GPU Reset Required）→ `AER: device recovery failed`。
被日志记为触发者的是 `pid=2696, name=python3`，即当时唯一在轮询 NVML 的
`/root/gpu-mon/gpumon.py` 监控脚本。**我的开启态在事件前约 36 秒启动，落在同一区间。**

**这台机器的 PCIe 链路在四卡重载下不稳定，而且这很可能不是偶然。**
三次掉卡的共同条件是"四卡 PCIe peer 密集流量"：第一次是别人跑的 80K benchmark，
第二次是我扫 FlashInfer vs NCCL 对照表，第三次是本 A/B 的开启态（pcie_ipc 正是
把这条链路用得更狠的那一侧）。**直接触发源我没有证据，不能归因于任何单一程序**，
但任何需要长时间四卡压测的方案都要按"链路可能掉"来设计。

**这直接改变了 pcie_ipc 这条路的评估口径**：它换来的收益在微基准里只有小消息档的
1.3–1.9 倍，而它恰好是**对这条已经证明不稳定的链路压力最大的实现**。
风险与收益要在同一个尺度上比，这条现在还没比出数。

---

## 7. 本会话四次掉卡的记录（2026-09-17）

**为什么要写这一节**：pcie_ipc 恰好是"四卡互相直写对方显存"最密集的实现，而本会话
四次掉卡都落在四卡 PCIe peer 密集流量期间。**这个相关性必须留成可核对的记录**，
不能只留在对话里——它直接决定这条路值不值得继续投。

| # | 时间（uptime 秒） | 当时在跑的负载 | 错误链 |
|---|---|---|---|
| 1 | 5073/5077（第一次 boot） | 早先的验证探针 `sm70_verify_bin` | Xid 43 |
| 2 | 57511（第二次 boot） | **别人的 80K benchmark**（`--input_tokens 80000`） | Xid 79 + 154，`pcieport 0000:04:04.0` `Multiple Uncorrectable`，四卡 `AER: can't recover`，`device recovery failed` |
| 3 | 1800（第三次 boot） | **我的 FlashInfer vs NCCL 四卡对照表**（`/tmp/fi_vs_nccl3`） | 同上，被日志记为触发者的是 `gpumon.py`（唯一在轮询 NVML 的进程） |
| 4 | 293（第四次 boot） | **我开 `FASTLLM_PCIE_IPC_AR=1` 跑引擎** | Xid 79 打在 05:00/07:00/08:00/09:00，四卡随后 Xid 154（GPU Reset Required） |

**共同点（观测）**：第 2–4 次都是同一个 `pcieport 0000:04:04.0`（PLX switch）报
`Multiple Uncorrectable (Non-Fatal)`，然后四张卡的 `nvidia` 驱动全报
`AER: can't recover (no error_detected callback)`。第 4 次发生在**机器重启后仅 293 秒**，
期间只跑过：一次 off 臂（干净通过）、一次 on 臂 A/B（通过）、然后这次 on 臂可达性验证。

**已知与未知，分开写**：
- **观测**：四次都在四卡 PCIe peer 密集流量期间；第 2–4 次是同一 switch、同一错误类型。
- **观测**：掉卡会被内核日志记为当时正在轮询 NVML 的进程（`gpumon.py`），
  **那不是触发者，只是当时在场的进程**。
- **未做**：没有做"A/B 交替重复看掉卡落在哪一臂"的对照（正在跑，见下）。
- **不能断言**：pcie_ipc 是掉卡的原因。**也没有证据说它与它无关。**

**正在做的判定**：交替重复 A/B（off/on 各 3 次交替，`/tmp/ab_rep.sh`）。
若 off 臂全部干净、掉卡只落在 on 臂，那就是强相关的证据；若两臂都掉，
说明与开关无关，是这台机器在四卡重载下的固有问题。
**对照结果（2026-09-17，已做）：与开关无关。**
- 会话 A：第 1 臂 `off`（OK）→ 第 2 臂 `on`（OK）→ 第 3 臂 `on`（**掉卡**）
- 会话 B：第 1 臂 `off1`（OK）→ 第 2 臂 `on1`（OK）→ 第 3 臂 **`off2`（掉卡，开关没开）**

两次的第 3 臂一个是开着 pcie_ipc、一个是关着，**所以掉卡与这个开关无关**；
它是"同一次开机里连跑第 3 个四卡基准"这个条件下的现象。这条规律已写进仓库
`AGENTS.md`（"一次开机最多连跑 2 个四卡基准"），做法改成**分多次开机、每次 1–2 臂**。

**在拿到这个对照之前，pcie_ipc 的引擎收益数字都不该采信**——掉卡会污染样本。

---

## 8. 掉卡判定：崩点定位与"偶发"结论（2026-09-17 晚）

### 8.1 崩掉那一次的日志证据（`/tmp/PC_1.out`，共 124 行）

命令里**只有** `FASTLLM_QWEN35_SM70_CUDA_GRAPH=0`，`FASTLLM_PCIE_IPC_AR` 等开关**都没设**。
失败顺序（按行号）：

| 行 | 内容 |
|---|---|
| 6 | 模型加载输出（`Load 0..100` / `Loading 0..100`，正常走完） |
| 16–19 | 四卡各一行 `FlashInfer attention disabled on GPU n (CC 7.0), using native attention` |
| 20–23 | 四行 `Native paged prefill uses chunked cublas attention` |
| **24** | **第一个失败**：`Error: CUDA error when synchronizing NCCL allreduce!` + `CUDA error = 999, cudaErrorUnknown`（`fastllm-multicuda.cu:3622`） |
| 27/32/35 | 同样的 NCCL 同步错误重复 |
| 38 | `[Fastllm] Model warmup failed: cublas error` |
| 41/44 | `Error: CUDA error when release memory on device 0!` |

**读法（计算 + 推断，分开写）**：
- **观测**：第一个报错**不是** cublas、也不是矩阵乘，而是 **NCCL 集合通信的同步**，错误码
  `999 = cudaErrorUnknown`。
- **推断（依据：999 的含义 + 后续调用全失败）**：卡在那一刻**已经不响应了**；
  cublas 报错、warmup 失败、释放显存失败都是**下游连带**，不是根因。
  **未验证**，因为我没法在掉卡瞬间抓现场（那一刻已无法再发起任何 CUDA 调用）。
- **旁观事实**：日志里 `warmup` 的位置正好是"加载完、刚要开始跑"的交接点，
  所以掉卡发生在**加载完成到首次计算之间**。

### 8.2 同一二进制、同一条命令，干净跑过 5 次

开机 12:43:42 那一次，用 `tools/gpu_crash_phase.sh` 连跑 5 次（`/tmp/PHASE_try1_1.summary`
与 `/tmp/PHASE_try2_{1..4}.summary`），**全部跑完、零掉卡**，`sha256` 全为 `adcb1bd7…`：

| 第几次 | Total (s) | 结束于开机后 |
|---|---|---|
| 1 | 3.2587 | 138 s |
| 2 | 3.2499 | 522 s |
| 3 | 3.2491 | 557 s |
| 4 | 3.2367 | 592 s |
| 5 | 3.2388 | 627 s |

**结论（观测）**：**同一个库、同一条命令、同一台机器，崩过一次也干净跑过 5 次
→ 掉卡不是确定性的软件行为。**

### 8.3 已被实测否掉的三种解释

| 曾以为的原因 | 反例 |
|---|---|
| 同一开机连跑第 3 次就崩 | 12:43 那次连跑 5 次全跑完；而 12:33 那次**第 1 次**就崩 |
| 与 `FASTLLM_PCIE_IPC_AR` / `PUSH4` 开关有关 | **关着开关也崩过**（8.1 那次命令里没有这些开关） |
| 与"开机后多少秒"有关 | 开机后 93 秒起跑跑完了，111 秒起跑崩了 |

**唯一还没验证的候选条件（推断，未验证）**：崩的那几次，另一个 agent 也在压同四张卡；
干净的那次只有我一个。**验证方法：同时起两个四卡负载，看是否必崩。**
在验证之前，这条不许写成结论。

### 8.4 撤错留痕

- 我先前写进 `AGENTS.md` 的"一次开机最多连跑 2 个四卡基准"**已被实测证伪**，
  已改写为"掉卡是偶发的、没找到触发条件、连跑多次允许、起跑前须确认无他人占用"。
- 我先前把掉卡归因为**硬件/链路问题**（依据是 `pcieport` 报 `Multiple Uncorrectable`）。
  **该归因已撤回**——按用户告知，硬件确定没问题。归因改为"未确定"。
- 我先前把"掉卡时开的那个开关"当成嫌疑（`pcie_ipc`、`push4`），**已被 8.3 的反例否掉**。

---

## 9. 代码复核（2026-09-17，自查；外部模型路由不可用，见撤错记录）

**复核方法**：按六类问题通读全部 475 行胶水层——跨 rank 一致性 / 就地改写 /
重入竞态 / 内存序 / 资源泄漏 / 非法访问。发现 4 处，全部已修并编译通过。

| # | 严重度 | 问题 | 修法 |
|---|---|---|---|
| 1 | 高 | 发射路径三个失败点（cudaSetDevice / staging 拷贝 / kernel 发射，原行 439/451/464）只 `return false`，**不设共享失败标志**。一个 rank 退回 NCCL、其余发射 pcie_ipc kernel → 集合通信不匹配 → 挂死 | 全部加 `s.failed.store(true)`；并引入 `inflight` 计数 + RAII 减量，发射前复核 generation |
| 2 | 高 | `Init` **先释放旧 slab、后 bump generation**，中间有窗口：通过 generation 检查的 rank 正要把已释放的指针交给 kernel | 顺序倒过来：先 bump、后等 `inflight==0`（上限 10 秒）、再释放 |
| 3 | 中 | `EnsureLocal` 的 `cudaSetDevice` 失败不设共享标志，其余 rank 在等待环空转 120 秒才一起退回 | 加共享失败标志 |
| 4 | 中 | `cudaMemset`（slab 首字节是信号槽）未查返回值 | 查失败并清退 |

**复核边界（诚实标注）**：只覆盖胶水层逻辑；上游 `pcie_ipc_all_reduce.cuh`（2344 行）
只读了协议约定，未逐行审。**这 4 处修复还没在显卡上验证过**（修完之后机器不可用）。

### 9.1 撤错：外部模型复核没做成

我曾表示“换一个模型复核”——实际路由 `ninerouter` 上可用的子代理模型与本会话
不兼容（child route 不允许），本地也没有可用的推理服务。**复核是我自己做的**，
不是外部模型做的。此条留痕，防止这条被误读成“已有独立复核”。
