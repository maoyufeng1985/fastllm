# FreeToken 论文（arXiv 2608.16157）阅读笔记：q* 策略、overlap 模型、全部量化结果

## 结论先说

1. 全文读到了，不是只有摘要。`https://arxiv.org/abs/2608.16157`（摘要页）和 `https://arxiv.org/html/2608.16157v1`（HTML 全文）都是 HTTP 200，两个都成功。没有失败的 URL。
2. q* 有闭式解，论文给得很清楚：`q* ≈ m · B_P / B_H`。优化的目标是**单层解码的暴露延迟**，也就是 `max(搬运时间, CPU 计算时间)` 里较大的那一支，不是吞吐。变量 m 是这一层这一步的**去重后缺失专家数**（既不在 GPU 缓存里、又没被别的专家顶掉的那些），q 是其中决定搬去 GPU 补齐缓存的数量，m - q 留在 CPU 上原地算。
3. 推导的关键假设只有一个：**PCIe 搬运和 CPU 专家计算抢的是同一份主机内存带宽**。所以链路跑满以后，CPU 只能吃到剩下的那点带宽。
4. 代码里 `recommend()` 的 2.0x 阈值**不是** q* 的推导，它是另一件事：一个粗粒度的后端开关（hybrid 还是 offload）[from the code]。q* 在代码里对应的是 `hybrid_fetch_fraction`，由 `load_hybrid_fetch_fraction()` 从实测的 overlap 带宽算出 [from the code]。
5. 论文**没有**独立的「q* 策略收益」消融实验，**没有**语义锚点 KV 缓存的量化消融，也**没有**在 NVLink / 数据中心 GPU 上做过任何评测。这三条都是缺口。

---

## 1. 内容来源：我实际抓到了什么

### 成功抓取的 URL

| URL | 结果 | 说明 |
| --- | --- | --- |
| `https://arxiv.org/abs/2608.16157` | HTTP 200 成功 | 摘要页。拿到标题、11 位作者、提交日期（2026-08-17，v1，cs.DC）、完整摘要、DOI `10.48550/arXiv.2608.16157` |
| `https://arxiv.org/html/2608.16157v1` | HTTP 200 成功 | HTML 全文。第一次通过工具抓取时输出被截断在 Table 1，我用 `curl` 重新下载了完整 HTML（168785 字节）并本地转成纯文本（61804 字符，1033 行），本节及以后所有「论文正文」内容均来自这份完整文本 |

### 失败的 URL

无。两个 URL 都成功，没有遇到 arXiv 被墙或只有摘要页的情况。

### 关于版本与附录的两点事实

- arXiv 上只有 v1，提交时间 2026-08-17 06:22:53 UTC，1,344 KB [from the paper text]。
- HTML 全文的章节编号只有 S1 到 S7，**没有 Appendix**。全文只有 **1 张数据表**（Table 1，测试机器），其余 5 个图表（Figure 1 到 Figure 5）都是图片。所有结果都以正文叙述和图注文字的形式给出，**没有第二张数值表** [from the paper text / my inference]。所以下文的量化表里，凡是标注「论文正文」的数，都是从叙述里读出来的，不是从机器可读表格里读出来的。这一点会影响复现的精度。

---

## 2. q* 策略：论文怎么推的

### 2.1 问题设定

术语先展开。一个 MoE 层（Mixture-of-Experts，混合专家层）里存着 E 个专家，每个 token 只被路由器（router）选中其中 k 个计算，k 远小于 E。发到 GPU 上的模型里，只有一部分专家常驻显存（VRAM），其余的留在主机内存（host memory，也就是 CPU 那侧的内存）。解码（decode）阶段每生成一个 token，都会有一批被选中的专家不在显存里，这就是「缺失」（miss）。

论文的符号 [from the paper text]：

- `M` = 这一层这一步需要处理、但不在缓存里的去重专家集合。
- `m = |M|` = 缺失专家的个数。
- `F` = 决定**搬运**到 GPU 缓存里、在 GPU 上算的那部分。搬进去之后还会留在缓存里供后续步骤复用。
- `C` = 决定**留在 CPU 上原地计算**的那部分，不改变缓存内容。
- `M = F ⊔ C`（不相交并集），`q = |F|`。
- `S` = 一个完整专家的字节数。
- `B_P` = 实测的 pinned 专家搬运带宽（host 到 device，走 PCIe）。pinned 指主机内存被锁页，可以给 DMA 直接搬，不用先拷一份。这是**搬运带宽**。
- `B_H` = 实测的 CPU 侧 MoE 专家核的有效带宽。这是**计算带宽**，量纲同样是 GB/s，因为 bs=1 小 batch 下专家计算是访存瓶颈，读多少字节决定多快。

### 2.2 被优化的目标

被优化的是**单层解码步的暴露延迟**（exposed latency），不是吞吐 [from the paper text]。

论文写得很直接：`The exposed layer latency is therefore the slower of the two concurrent branches, which is precisely the quantity that Equation 4 balances.` 即暴露延迟 = 两条并发分支里较慢的那一支：

```
T_exposed = max(T_fill(q), T_cpu(m - q))
```

因为两条分支同时在跑，谁慢谁决定这一层的耗时。所以目标函数是 min-max，不是求和，也不是最大化吞吐。

### 2.3 带宽项与残差带宽论证

论文的推导链条 [from the paper text]：

**第一步：说清两条分支在抢同一份资源。**
> `Since both expert DMA transfers and CPU execution read from the same host-memory subsystem, a saturated PCIe transfer leaves a residual bandwidth of:`

**第二步：定义残差带宽 B_R（论文式 2）。**

```
B_R = max(B_H − B_P, 0)
```

意思很直白：主机内存总共能供 B_H 这么多字节每秒。PCIe 搬运已经吃掉了 B_P。剩下给 CPU 专家计算的，就是差的那部分。**这就是 q* 策略成立的物理前提**：假设 CPU 的专家计算本来能吃满 B_H，被 DMA 抢走 B_P 之后只剩 B_H − B_P。

**第三步：写两条分支的时间（论文式 3）。**

```
T_fill(q)  ≈ q·S / B_P
T_cpu(m−q) ≈ (m−q)·S / (B_H − B_P)
```

注意 T_cpu 的分母是 `B_H − B_P`，不是 `B_H`。这就是残差带宽的用法。

**第四步：令两支相等，解比例（论文式 4）。**

```
q / (m − q) ≈ B_P / (B_H − B_P)
```

推出闭式解：

```
q* ≈ m · B_P / B_H
```

也可以写成 `q*/m = B_P/B_H`：缺失里应该有**实测搬运带宽占实测 CPU 带宽的比例**那么多去走 PCIe，剩下的留 CPU。移动端常见情况是 B_P（约 11.8 到 52.7 GB/s）远小于 B_H（约 47.5 到 178 GB/s），所以 q*/m 通常很小，多数缺失应该在 CPU 上算。

### 2.4 退化情形

论文明确讲了一个退化情形 [from the paper text]：

> `As B_H approaches B_P, q* approaches the total miss count m, and the system degenerates into pure on-demand cache fill without requiring separate execution branches or policies.`

翻译：当 CPU 带宽掉到跟 PCIe 带宽差不多的时候，q* 就接近 m，也就是全部搬过去、全部在 GPU 上算，系统退化成「纯按需填缓存」，根本不需要 CPU 分支这条路。这解释了为什么不需要一个单独的开关：公式自己会在两个极端之间连续过渡。

### 2.5 算法细节（论文给了，但不是伪代码）

论文没有给伪代码，只给了这几条实现约定 [from the paper text]：

1. `q*` 取整（rounds to an integer）。
2. `F` 具体选哪几个专家，交给缓存替换策略（LRU）决定，不由带宽模型决定。
3. **永远至少保留一次补齐**（`It always retains at least one fill`），这样即使大部分缺失由 CPU 处理，缓存还在继续变热。
4. `B_H` 和 `B_P` 在部署时对目标硬件实测得到，不是从规格书读的。
5. 执行顺序：先发 CPU 分支；再跑 GPU 缺失路径（更新缓存、批量拷贝 F、对合并后的 GPU 计算集合 `G = H ∪ F` 做分组求值，H 是缓存命中的那批）；CPU worker 并发处理 C。
6. CPU 和 GPU 各算各的部分和，最后合并。合并是精确的，没有算法近似（`preserving the exact MoE output without algorithmic approximation`）。

论文 Figure 2 的图注给了一个具体算例 [from the paper text]：12 个被路由到的专家里 8 个命中缓存，`m = 4` 个缺失，`q* = m·B_P/B_H` 的结果是搬运 1 个、CPU 原地算 3 个。

### 2.6 对比代码实现（`python/freetoken/moe/benchbw.py`）

仓库在 `/tmp/FreeToken`，commit `cac247a`（`chore(release): 0.1.3 (#489)`）。

**先说结论：代码把论文的闭式解实现成了 `hybrid_fetch_fraction`，数值上是同一个式子；`recommend()` 的 2.0x 阈值是另一层粗筛，论文里没有。**

#### (a) 2.0x 阈值确实不是 q* [from the code]

```python
def recommend(cpu_bw_gbs: float, pcie_bw_gbs: float, threshold: float = 2.0) -> str:
    """``hybrid`` iff CPU bandwidth exceeds ``threshold`` x PCIe bandwidth, else ``offload``."""
    return "hybrid" if cpu_bw_gbs > threshold * pcie_bw_gbs else "offload"
```

这是一个**二值后端选择**：CPU 带宽超过 PCIe 带宽 2 倍，就用 hybrid 后端（CPU 参与算专家）；否则用 offload 后端（全部走 PCIe 搬到 GPU 算）。它用在独立测量（standalone）的数上，不用于 overlap 的数 [from the code]。

论文里跟它最接近的一句话是退化情形的描述（B_H 接近 B_P 时退化成纯按需补齐），但**论文没有给 2.0 这个系数**，也没有说它是一个硬阈值 [from the paper text / my inference]。所以这个 2.0x 是工程上的安全余量，不是论文推导的产物。

顺带一个保守化设计 [from the code]：`load_backend_recommendation()` 要求同一格式下**所有**被 bench 的 workload 都投票 hybrid 才返回 hybrid，只要有一个返回 offload（说明该格式卡在阈值附近）就整体降级成 offload。代码注释直接写了理由：`a mixed verdict (a near-threshold format) resolves conservatively to "offload"`。

#### (b) `measure_overlap_bw` 就是在测论文式 3 的两个分母 [from the code]

```python
def measure_overlap_bw(fmt, wl, device, num_threads=0, seconds=2.0) -> dict:
    """Concurrent achieved bandwidths (GB/s): the CPU MoE GEMV and the PCIe gather running
    at the same time -- the contention regime hybrid decode's overlap actually lives in.
    ...
    """
```

它做的事：起一个 worker 线程死循环跑 bs=1 的 CPU MoE decode 步（`run_task` 会释放 GIL，所以是真并发），主线程同时死循环跑整层的 `copy_missing()` gather，每次 copy 后同步以保证 DMA 真的在飞而不只是入队。两边用一个 barrier 对齐起跑，各跑 `seconds`（默认 2.0）秒，各自汇报 `bytes / 自己的耗时`。返回 `{"cpu_gbs": ..., "pcie_gbs": ...}`。

代码注释对为什么要实测而不是算，说得很清楚 [from the code]：

> `Assuming full DRAM contention (CPU keeps cpu_bw - pcie_bw under DMA) over-penalizes a CPU kernel that never saturated DRAM to begin with -- the DMA then mostly rides the leftover bandwidth; assuming no contention ignores it entirely.`

翻译：论文式 2 的「CPU 只拿到 B_H − B_P」是一个假设，代码作者认为这个假设可能**过度惩罚** CPU 那侧（如果 CPU 核本来就没吃满内存带宽，DMA 其实是搭了顺风车）。所以代码直接测争用状态下的实际值，而不是靠假设去减。

#### (c) 从实测带宽到 fraction 的换算 [from the code]

```python
def load_hybrid_fetch_fraction(...) -> float | None:
    for entry in entries:
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            return min(1.0, pcie / cpu)
    return None
```

两条路：

- **首选（overlap 实测）**：`fraction = pcie_ov / (pcie_ov + cpu_ov)`。
- **回退（只有独立测量时的旧 profile）**：`fraction = pcie / cpu`。

**跟论文对不对得上？** 数学上对得上 [my inference]。把论文的理想争用假设代进首选式：ideal 情况下 `cpu_ov = B_H − B_P`，`pcie_ov = B_P`，于是

```
pcie_ov / (pcie_ov + cpu_ov) = B_P / (B_P + B_H − B_P) = B_P / B_H
```

正好就是论文的 `q*/m`。而回退路径 `pcie/cpu = B_P/B_H` 也正好是论文式。所以两套公式是**同一个目标比例**的两种取数口径：论文用「未争用的 B_H 减掉 B_P」推残差，代码用「争用下实测的 cpu_ov 和 pcie_ov」直接量。代码注释也是这么说的，说回退路径在「满 DRAM 争用假设下」退化成 `pcie/cpu`。

差异在哪 [my inference]：论文 Table 1 里的 B_H 是**未争用**的独立测量值（比如 5090 服务器上 77.3 GB/s）；代码首选的 `pcie_ov + cpu_ov` 是**争用**后的。同一台机器上两者不等，所以最终 fraction 的数值会不一样，但设计意图一致。代码这条路径更保守也更贴近实际，因为它不依赖「CPU 核能吃满 B_H」这个较弱的假设。

#### (d) 取整规则 [from the code]

论文只说「rounds q* to an integer」。代码里的规则是**最小化较慢那一支**，不是四舍五入也不是向上取整：

```python
lo = (num_missing * fetch_frac_q16) >> 16
cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
cost_hi = tl.maximum((lo+1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16)
max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
```

用 Q16 定点数（把 fraction 乘以 2 的 16 次方取整）保证 GPU 核和 CPU 参考实现算出**逐位相同**的结果，也保证 CUDA graph 里没有浮点不确定性和主机同步 [from the code]。

测试里有一条专门记录了这个规则修过的 bug [from the code]：

> `ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.`

翻译：3 个缺失、比例 41.5%，向上去整会取 2，这会让 PCIe 那支比平衡点慢约 1.6 倍；规则会正确地取 1。

#### (e) 一处代码与论文文字对不上 [from the code / my inference]

论文写「It always retains at least one fill」（永远至少补齐一次）。但我在设备端核（`_ensure_experts_hybrid_kernel`）和 CPU 参考实现（`_ensure_experts_hybrid_cpu`）里都只看到

```python
num_fetch = tl.minimum(num_missing, max_fetch)   # GPU
num_fetch = min(len(missing), int(max_fetch))    # CPU 参考
```

**没有**显式的「至少 1」钳位。当 `m = 0` 时 `num_fetch = 0` 是合理的（本来就没缺失）；当 `m ≥ 1` 而 fraction 很小时，min-max 规则会不会必然取到 ≥ 1，我没有构造出反例也没有证实，这里只如实报告「未在代码中找到论文所述的那条钳位」。这一条标 [my inference]：可能是论文描述与当前 commit 的实现有出入，也可能是我没找全。

#### (f) 选哪几个缺失去补齐 [from the code]

论文说 `F` 的选择交给缓存替换策略。代码里是一个叫 `BY_RECENCY` 的策略（默认开）：优先补齐「本步之前最近被激活过」的缺失专家，用 `expert_recency` 做 LRU，平局取较小的专家 id。注释说明动机是 `this prioritizes *recurring* misses for caching, lowering the steady miss rate`，即优先缓存会反复出现的缺失，从而压低稳态缺失率。代码里保留了一个老策略作为对照：取最小的 expert id，注释称之为 `the original routing-blind heuristic`（路由无关的原始启发式）。

驱动器侧的接线 [from the code]：`engine.py` 的 `_resolve_hybrid_fetch()` 负责把命令行参数 `--moe-hybrid-max-fetch -1`（auto）解析成 fraction。解析成功时把 `cache.hybrid_max_fetch` 设成 `num_experts`（注释写 `inert: the fraction is the cap`，即让固定上限失效，改由 fraction 控制），并打一条 info 日志报告取到的百分比。解析失败（没有可用 profile）时退回固定上限 1，并打 warning。

---

## 3. overlap 模型：论文怎么讲计算与搬运的重叠

### 3.1 两个阶段、两套 overlap 机制

论文的阶段划分 [from the paper text]：

- **Prefill（预填充）**：处理输入提示词。这一步要一次算几千个 token，路由覆盖了几乎全部专家，所以工作集实际上变「稠密」。
- **Decode（解码）**：一次生成一个 token。这一步只有稀疏的专家被激活，但缓存缺失要反复补齐。

### 3.2 Prefill 阶段：整层双缓冲（full-layer double buffering）

谁跑在哪个引擎上 [from the paper text]：

- GPU 上：当前层 l 被路由到的那些专家的计算。
- 一条**专用的搬运流**上：下一层 l+1 的**整层**专家集合，通过 PCIe 搬进来。

关键设计点：**搬的是整层，不是按需搬被路由到的那几个**。原因是预填充阶段路由覆盖几乎全部专家，提前知道路由没有意义；搬整层的好处是搬运可以在该层路由结果出来**之前**就开始，于是权重移动在后台连续进行，不会在层与层之间串行化。

缓冲区共享 [from the paper text]：两块整层的缓冲区都从全局的 slot pool（槽位池）里分配，跟解码缓存是**同一个池**。所以没有单独的 prefill 缓存，也没有阶段切换（phase handoff），预填充结束时还留着的条目会直接给延迟敏感的解码阶段用。

瓶颈资源 [from the paper text]：论文的说法是 `Full-layer double buffering makes prefill transfer-bound`，也就是搬运受限，专家计算被完全藏在搬运背后。给的上界非常具体：

> `with overlap on, each 8,192-token prefill chunk completes in 1.19–1.22 s, the time to stream the 64.4 GB expert pool once at 52.7 GB/s—the practical ceiling of the PCIe 5.0 ×16 link`

翻译：每 8,192 token 一个预填充块，1.19 到 1.22 秒跑完，这个时间正好等于把 64.4 GB 的专家池在 52.7 GB/s 下流一遍的时间，而 52.7 GB/s 就是 PCIe 5.0 x16 链路的实际上限。所以可达 overlap 的**界**是：

```
T_chunk ≥ 专家池字节数 / B_P
```

只要计算时间不超过这个值，就完全被藏住。论文接着说 `so expert computation is fully hidden behind transfer`，吞吐在 16k token 时爬到 6.7k tok/s。

降级路径 [from the paper text]：当槽位池腾不出两个整层时，FreeToken 退回按需加载预填充专家，而不是超额订阅显存（`falls back to on-demand prefill loading rather than oversubscribing GPU memory`）。

### 3.3 Decode 阶段：按层做带宽匹配的分割

这一阶段的 overlap 是「同一层内、两条分支并发」，不是「层间流水」[from the paper text]。

- 芯片划分：
  - GPU：路由（router）和缓存查找、缓存命中集 `H` 的求值、被选中搬运集 `F` 的求值，以及缓存更新。合并集合记为 `G = H ∪ F`，一次分组求值（grouped evaluation）。
  - CPU：留在主机内存的专家集合 `C`，原地计算。用持久化的 C++ worker 池，绑到物理核上，用架构相关 SIMD 加核内反量化（in-kernel dequantization），保证这条路径是访存受限的，返回按门控权重加权的 per-token 部分输出。
- 谁先发：**CPU 分支先发**，然后才跑 GPU 缺失路径（缓存更新、批量拷贝 F、分组求值 G）。
- 瓶颈资源：主机内存带宽。因为 PCIe 搬运和 CPU 计算共享同一个主机内存子系统，PCIe 跑满后 CPU 只能吃残差。
- 可达 overlap 的模型，就是式 3 和式 4：暴露的层延迟 = `max(T_fill(q), T_cpu(m−q))`，而 q* 正是令这一 max 最小的那个点。

### 3.4 在图上跑（这部分是必要条件，不是优化）

论文明确说这套 per-layer 控制流（缺失检测、集合定大小、受害者选择、CPU 分支本身）要塞进一个静态捕获的 CUDA Graph（CUDA 图，把一系列核和拷贝录下来一次性重放的机制），本身就是一个独立实现难题 [from the paper text]，并在 §4.1 讲了解决方案。

关键约束 [from the paper text]：**所有跟路由相关的决策必须留在 GPU 上，以「数据」的形式存在于静态图里，而不是由主机控制**。因为缺失哪些专家、要取几个、淘汰谁，每一步都在变；主机控制会在每个 MoE 层插一次昂贵的设备同步。

---

## 4. 全部量化结果

重要提醒：**以下所有数值来自论文正文叙述和图注文字，不是机器可读表格**。论文只有 1 张数据表（Table 1），其余都是图片。这就是为什么「配置」一栏经常只能写到模型和部分机器设置。

### 4.1 Table 1：六台测试机器（这是全文唯一的数值表）

带宽都是在部署所用的张量形状上实测的，不是从平台规格书读的。三台租用服务器上，CPU 线程数和 DRAM 是容器配额。B_P = 实测的 host 到 device 专家搬运带宽（走 PCIe），B_H = 实测的 CPU 侧 MoE 专家核有效带宽。

[from the paper text]

| 机器 | GPU（显存） | PCIe | B_P (GB/s) | CPU（线程） | DRAM (GiB) | B_H (GB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 5090（服务器） | RTX 5090 (32 GB) | 5.0 ×16 | 52.7 | 2× Xeon Gold 6459C (32) | DDR5 180 | 77.3 |
| 4090 | RTX 4090 (24 GB) | 4.0 ×16 | 25.1 | 2× Xeon Platinum 8358P (32) | DDR4 240 | 63.2 |
| 3090 | RTX 3090 (24 GB) | 4.0 ×16 | 25.3 | 2× Xeon Gold 6330 (28) | DDR4 180 | 56.7 |
| 5090 desktop | RTX 5090 (32 GB) | 5.0 ×16 | 49.0 | Ryzen 9 9950X3D (32) | DDR5 192 | 53.8 |
| 4060 laptop | RTX 4060 Laptop (8 GB) | 4.0 ×8 | 11.8 | Core i9-13900H (20) | LPDDR5 32 | 47.5 |
| PRO 6000 | RTX PRO 6000 (96 GB) | 5.0 ×16 | 51.5 | Xeon Platinum 8559C (48) | DDR5 512 | 178 |

**一处论文内部不一致，如实记录**：正文说三台租用服务器被限制在 6 个 CPU 线程并绑到 GPU 的 NUMA 节点，这样交付 `56.7–77.3 GB/s` 的主机带宽，与两台真机在满线程下达到的量级相同（`53.8 GB/s on the desktop's 16 cores, 47.5 GB/s on the laptop's 14`）[from the paper text]。这里正文写的 16 核和 14 核，与 Table 1 里写的 32 和 20（列名是「CPU (threads)」）对不上。我无法判断哪个是笔误，只报原文。

### 4.2 主结果：RTX 5090 上的端到端（Figure 3）

配置 [from the paper text]：RTX 5090；Qwen3.6-35B-A3B BF16 用 6 个 CPU 线程，DeepSeek-V4-Flash MXFP4 用 8 个。四个 agentic workload，引擎自己的 harness。

指标定义 [from the paper text]：decode 吞吐 = 每请求平均 tok/s；TTFT（time-to-first-token，首个 token 延迟）= 每请求平均。论文明确说 `Agent trajectories diverge across engines so cross-engine wall-clock totals are not compared`，也就是不比较跨引擎的总墙钟时间。

| 指标 | 数值 | 硬件 | 模型 | 配置 |
| --- | --- | --- | --- | --- |
| decode 吞吐 | 77 到 83 tok/s | RTX 5090 | Qwen3.6-35B-A3B BF16 | 6 CPU 线程 |
| decode 吞吐相对最强基线 | 1.8x 到 2.3x | RTX 5090 | Qwen3.6-35B-A3B BF16 | 同上 |
| decode 吞吐 | 22 到 25 tok/s | RTX 5090 | DeepSeek-V4-Flash MXFP4 | 8 CPU 线程 |
| decode 吞吐相对最强基线 | 1.5x 到 1.9x | RTX 5090 | DeepSeek-V4-Flash MXFP4 | 同上 |
| agentic 稳定性 | 相对单轮 W1 值波动在 12% 以内 | RTX 5090 | Qwen3.6 | 三个 agent workload |
| 对照：KTransformers 在 W2 的衰减 | 已从 W1 掉 31% | RTX 5090 | DSV4-Flash | Ditto |
| MoE-Infinity 可服务的 workload | 仅 W1，8.8 tok/s | RTX 5090 | [未取到具体模型] | 它按专家预填充分级的上限会中断更长提示词的 workload，且自带 server 跨请求不保留 KV 缓存 |
| TTFT 最低 | 六个多轮 cell 里的五个 | RTX 5090 | 两个模型 | Qwen3.6 × W3 这一格归 KTransformers 的 GPU-prefill 分支；W1 的短隔离提示词归 llama.cpp |
| FreeToken 最差单轮 TTFT | 每个 cell 都低于 44 s | RTX 5090 | 两个模型 | [未取到各 cell 的具体均值] |
| 各基线越过的 TTFT 上界 | llama.cpp 232 s、Ollama 179 s、KTransformers 946 s | RTX 5090 | [未取到逐项对应模型] | 每台引擎在至少一处超过 150 s |

补充 [from the paper text]：OpenClaw 自带 120 s 空闲看门狗，Claude Code 默认请求超时约十分钟。所以论文把尾部 TTFT 定性为 `an availability boundary, not a latency statistic`（可用性边界，不是延迟统计量）。

**注意**：论文摘要和引言里给的是 `1.5–2.3x` 的总口径 [from the paper text]。§5.2 拆开给的是 Qwen3.6 的 1.8 到 2.3 倍和 DSV4-Flash 的 1.5 到 1.9 倍。两个说法都在原文里，不冲突，只是粒度不同。

### 4.3 消融一：流水化预填充，也就是双缓冲的收益（Figure 4a）

配置 [from the paper text]：RTX 5090，Qwen3.6-35B BF16，预填充 TPS 对提示词长度。这就是**唯一一个隔离 overlap 收益的消融实验**。

| 指标 | 数值 | 硬件 | 模型 | 配置 |
| --- | --- | --- | --- | --- |
| 开 overlap 时每个 8192-token 预填充块耗时 | 1.19 到 1.22 s | RTX 5090 | Qwen3.6-35B BF16 | 等于 64.4 GB 专家池在 52.7 GB/s 下流一遍 |
| 关掉第二块缓冲区（去掉双缓冲）的代价 | 4k token 时 −19%，8k 时 −25%，16k 时 −26% | RTX 5090 | Qwen3.6-35B BF16 | 变成搬运与计算串行 |
| 吞吐在 16k token 时 | 6.7k tok/s | RTX 5090 | Qwen3.6-35B BF16 | 开 overlap |
| 专家池体积 | 64.4 GB | RTX 5090 | Qwen3.6-35B BF16 | [未取到是不是全池，按上下文是] |

论文对趋势的解释 [from the paper text]：惩罚随提示词长度增长，因为被隐藏掉的计算占比在上升。

### 4.4 消融二：专家缓存局部性（Figure 4b）

配置 [from the paper text]：把 W1 到 W4 四个 workload 的**相同路由 trace** 重放到三种放置策略上，缓存容量对齐。线是 W1 到 W4 的均值，带是 min 到 max 范围。这是**专家缓存的消融**，方式是对比不同策略的缺失率，而不是开关缓存。

| 指标 | 数值 | 硬件 | 模型 | 配置 |
| --- | --- | --- | --- | --- |
| RTX 5090 的服务容量 | Qwen3.6 专家池的 37%，DSV4-Flash 专家池的 11% | RTX 5090 | 两个模型 | 等容量对比 |
| FreeToken 全局 LRU 的解码期专家读缺失率 | 16%（Qwen3.6）与 39%（DSV4-Flash） | RTX 5090 | 同上 | 37% / 11% 容量 |
| KTransformers 预填充时更新的放置 | 41% 与 59% | RTX 5090 | 同上 | 等容量 |
| llama.cpp 路由无关的静态切分 | 62% 与 89% | RTX 5090 | 同上 | 等容量 |

论文补充 [from the paper text]：`The ordering holds across workloads at every capacity short of the full pool.` 也就是只要缓存没大到能装下整个专家池，这个排序在所有 workload 和所有容量上都成立。

### 4.5 消融三：跨硬件（Figure 5）

配置 [from the paper text]：W2 编码 agent（SWE issue，走 OpenCode harness），Qwen3.6-35B-A3B。4060 laptop 用 NVFP4，其余 Qwen3.6 列用 BF16。RTX PRO 6000 那一列是另一个演示：GLM-5.2（753B-A40B，NVFP4）跑数学 workload，Ollama 没在那里跑。叉号表示某引擎无法服务。

| 指标 | 数值 | 硬件 | 模型 | 配置 |
| --- | --- | --- | --- | --- |
| 相对最强基线的领先倍数 | 1.3x | RTX 3090 与 RTX 4090 | Qwen3.6-35B-A3B | W2 |
| 同上 | 1.9x | RTX 5090（服务器） | Qwen3.6-35B-A3B | W2 |
| 同上 | 2.1x | RTX 5090 desktop | Qwen3.6-35B-A3B | W2 |
| 同上 | 1.8x | RTX 4060 laptop | Qwen3.6-35B-A3B NVFP4 | W2 |
| 4060 laptop 绝对速度 | 39.3 tok/s | RTX 4060 Laptop 8 GB，PCIe x8 | Qwen3.6 NVFP4 | 达到 RTX 4090 速率的 92% |
| 服务器换到桌面主机（同为 5090）的损失 | FreeToken 掉 4% | RTX 5090 服务器 vs 桌面 | Qwen3.6 | 从多通道服务器换到双通道消费桌面 |
| 同场景 llama.cpp 的损失 | 只保住 80% 的速率 | 同上 | 同上 | 论文归因于它的 CPU 常驻专家在两条 DDR5 通道上饿死 |
| GLM-5.2 吞吐 | FreeToken 14.9 tok/s，llama.cpp 7.3 tok/s，即 2.0x | 单张 RTX PRO 6000 | GLM-5.2 753B-A40B，NVFP4 | 专家权重逐位相同；平均 TTFT 7.5 s vs 7.8 s |
| KTransformers 在该机器的可服务性 | 无可用路径 | RTX PRO 6000，512 GiB 主机内存 | GLM-5.2 | 它的 GLM-5.2 方法需要 753 GB 到 1.5 TB 的主机常驻专家，超出 512 GiB；CPU 核也读不了 GLM-5.2 的 NVFP4 布局 |

### 4.6 背景量化数字（引言、挑战分析、Figure 1）

| 指标 | 数值 | 来源位置 | 备注 |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash 参数 | 284B 总参，13B 激活；43 层，每层 256 个路由专家里激活 6 个 | 引言 | 部署精度下激活参数fit进 RTX 5090 的 32 GB |
| DSV4-Flash FP4 部署的专家权重搬运量 | 约 140 GB | §2.1 | 每次预填充几乎要流完整池 |
| 该搬运量在 RTX 5090 系统上的时间 | 约 2 s | §2.1 | 论文写 PCIe 5.0 x16，约 60 GB/s（这是论文给的链路口径，与 Table 1 实测 52.7 不同） |
| 在 4090/3090 级桌面 | 约 5 s | §2.1 | 论文写 PCIe 4.0 x16，约 25 GB/s |
| 在笔记本常见的 x8 链路上 | 10 s 或更多 | §2.1 | |
| 从 7 GB/s 的 NVMe 读 140 GB 专家池的启动成本 | 约 20 s，还没算预热 | §2.3 | |
| RTX 5090 的稠密 BF16 吞吐相对 | 约为 H100 的 1/5，B200 的 1/10 | §2.1 | 用来论证消费级 GPU 藏不住重复预填充 |
| 双通道 DDR4 峰值带宽 | 约 50 GB/s | §2.2 | 消费平台只有两个 DRAM 通道 |
| 双通道 DDR5 峰值带宽 | 80 到 90 GB/s | §2.2 | |
| 单张 RTX 4090 或 5090 从板载显存取的带宽 | 1 到 1.8 TB/s | §2.2 | 作为对比 |
| Codex 生产 trace 的解码速度中位数 | 33 tok/s | Figure 1b，引自 Zhu et al. 2026 | 论文用作「可交互」的基准线 |
| 8 GB RTX 4060 laptop 上的 35B 模型 | 39.3 tok/s，超过 33 tok/s | 摘要与 Figure 1 | |
| 32 GB 游戏桌面上的 284B 模型 | 可交互服务 | 摘要 | [未取到具体 tok/s] |
| 支持的 MoE 模型数量 | 20 个以上 | 摘要与结论 | 评测只做了 3 个 |
| 评测使用的模型数 | 3 个：Qwen3.6-35B-A3B、DeepSeek-V4-Flash、GLM-5.2 | §5.1 | |
| Kimi-K3 开放权重体积 | 594 GB，超出消费级内存 | Figure 1 图注 | |
| Steam 月活 | 2 亿以上，约 72% 的调查系统有独立 NVIDIA GPU | 引言 | |
| Claude Code 企业部署成本 | 每个开发者每活跃日约 13 美元，每月 150 到 250 美元 | 引言脚注（引自 Anthropic 2026） | |

### 4.7 明确**没有**报告的消融

这些是任务点名要查的，我如实报告结果为「未取到」：

| 想要的消融 | 论文里有没有 | 结论 |
| --- | --- | --- |
| 单独隔离 overlap 的收益 | 有 | Figure 4a 双缓冲开关，−19% / −25% / −26% |
| 双缓冲的收益 | 有 | 同上，这是同一个实验 |
| CPU-GPU 分割（q* 策略）的收益 | **没有** | 论文只给了闭式推导和 Figure 2 的一个示意算例（m=4，搬 1 算 3），**没有** q 值扫描、没有「固定 q 对比 q*」的实测对比。未取到 |
| 专家缓存的收益 | 部分有 | Figure 4b 是等容量下三种放置策略的缺失率对比（16%/39% vs 41%/59% vs 62%/89%）。**没有**缓存大小的性能扫描曲线数值，只有缺失率曲线 |
| 语义锚点 KV 缓存的收益 | **没有** | §3.1 详细描述了机制（在思考片段、工具调用、工具输出、对话轮次这些特殊 token 边界上打检查点），但**没有任何量化实验**。未取到命中率、未取到省下的预填充时间、未取到消融 |
| 弹性显存重配（运行时重建缓存）的收益 | **没有** | §3.3 描述了机制，无私测 |
| 快速启动（FTW 格式、先读后 pin）的收益 | 只有间接数 | 只有「系统读 140 GB 需约 20 s」这个背景数，**没有** FreeToken 自己启动时间的实测对比 |

---

## 5. 论文陈述的局限与假设

论文**没有独立的 Limitations 章节** [my inference]。以下是从正文里抽出来的、论文自己写明的假设，以及我推断在别处会失效的地方。逐条标注。

### 5.1 论文自己写明的假设

1. **PCIe 搬运和 CPU 专家计算共享同一份主机内存带宽** [from the paper text]。原文：`Since both expert DMA transfers and CPU execution read from the same host-memory subsystem`。这是 q* 推导最核心的前提，也是我判断最脆弱的一条。
2. **B_H、B_P 在部署时对目标硬件实测** [from the paper text]。不是从规格书读的，论文特别强调 `All bandwidths in Table 1 are measured on the deployed tensor shapes rather than taken from platform specifications`。
3. **预填充是搬运受限的** [from the paper text]。整层双缓冲让专家计算完全藏在搬运背后，所以预填充块时间的下界是专家池字节数除以 B_P。
4. **显存不足时的两条降级路径** [from the paper text]：槽位池腾不出两个整层时，退回按需加载预填充专家（而不是超额订阅显存）；整个专家池无法被 pin 或注册 DMA 时（论文说这在某些操作系统和驱动配置上是限制），退回纯 CPU 的 MoE 后端，专家权重留在可分页的主机存储里，所有路由专家在 CPU 上算，非专家层仍在 GPU 上，只有激活大小的输入、路由元数据和聚合输出跨 CPU-GPU 边界。论文自己承认这条路径 `trades peak transfer bandwidth for deployability`。
5. **正确性与显存无关** [from the paper text]。原文：`because the CPU-resident expert pool remains the source of truth, GPU memory affects only performance, never correctness`。这是运行时改显存预算的前提。
6. **不改变模型、不损失精度** [from the paper text]。CPU 和 GPU 各算部分和再合并，`preserving the exact MoE output without algorithmic approximation`。相关的对照是：HOBBIT 取降精度副本，SiDA 和 SMoE 替换或跳过低分专家，Pre-gated MoE 改路由并微调。FreeToken 声称保持路由计算精确、模型不改。
7. **CPU 侧的评估是模拟的** [from the paper text]。3090、4090、5090 这三台是租用的双路服务器，CPU 远超任何边缘主机，所以所有服务运行和带宽测量都被限在 6 个 CPU 线程并绑到 GPU 的 NUMA 节点。论文用这种方式让服务器的交付带宽落到 56.7 到 77.3 GB/s，与真机的 53.8 和 47.5 同量级，并用 desktop 和 laptop 两台真机来验证这个模拟。
8. **agent 轨迹因引擎而异，所以不比总墙钟时间** [from the paper text]。只比每请求平均的吞吐和 TTFT。
9. **评测用的 workload 是脚本化的** [from the paper text]。W2 是三个脚本化用户轮次，W4 是十三个固定用户轮次，且 W4 把 OpenClaw 的 120 s 空闲看门狗关掉了，好让慢引擎也能被测到。

### 5.2 我推断在 NVLink / 数据中心 GPU 上不成立的

全部标 [my inference]：

1. **q* 的前提直接崩掉。** 推导的物理依据是「PCIe 搬运和 CPU 计算读同一份主机内存」。在 NVLink / NVSwitch 的数据中心配置下，GPU 之间的通信不走主机内存，CPU 也不是承载专家池的地方（显存足够大，专家根本不用放主机内存）。整条残差带宽论证 `B_R = B_H − B_P` 就没有意义了。这个策略不是「在 NVLink 上收益变小」，而是**没有可应用的场景**。
2. **B_P 这个量本身就是 PCIe 的。** 论文 Table 1 里 B_P 的列标题就是 PCIe 链路，取值 11.8 到 52.7 GB/s。NVLink 的量级高一个数量级（几百 GB/s 到 TB/s 级），`q*/m = B_P/B_H` 会趋近 1，也就是退化成「全部搬过去、全部在 GPU 上算」的那个退化情形。论文确实提到这个退化方向（B_H 接近 B_P 时 q* 接近 m），但它是从 B_H 变小的方向叙述的，不是从 B_P 变大的方向。
3. **整层双缓冲的内存预算假设不成立。** 双缓冲要求从槽位池里腾出**两个整层**的专家。桌面 32 GB 显存下 64.4 GB 的池意味着一个整层约占相当比例，需要精打细算；48 GB 到 96 GB 的数据中心卡上这个约束基本不存在，双缓冲的收益也就无从体现（本来就不需要从主机搬）。
4. **主机分配的显存预算波动这条动机不存在。** §2.3 整个论证的前提是「边缘设备上 GPU 被桌面合成器、浏览器、游戏共享，显存随时可能被抢」。数据中心里这个前提不成立，所以弹性显存重配（§3.3 的运行时重建缓存）在那种环境里没有动机。
5. **评估的硬件全是消费级 x8 到 x16 的 PCIe 和双通道内存。** 论文没有在任何 NVLink 机器上跑过一个数。这一点是事实陈述，不是推断 [from the paper text]。
6. **6 线程封顶这个模拟假设的边界。** 论文用「限制到 6 线程让服务器带宽掉到消费级量级」来模拟边缘主机。这在带宽维度上看起来合理，但 CPU 微架构的差异（比如有没有 AMX、有没有 AVX-512 BF16）被这个模拟抹掉了，而 B_H 恰恰取决于这些。 [my inference]

### 5.3 我推断在非 MoE 模型上不成立的

全部标 [my inference]：

1. **整个框架依赖稀疏激活。** 论文自己在引言里把 MoE 的两面说清楚了：稀疏激活让计算变得可行，但完整专家池让服务变得困难。对于稠密（dense）模型，没有「专家」这个东西，没有专家池，没有按 (layer, expert) 的驻留，没有每 token 路由变化，所以缓存局部性这条（§3.2 的语义感知专家缓存）和带宽自适应这条（q* 策略）都失去了作用对象。
2. **两层的显存层级结构消失。** FreeToken 的组织方式是「CPU 常驻专家池是真理之源，GPU 剩余显存是弹性专家缓存」。稠密模型不存在这个可以分层的、按需取用的池，只能整体放进显存或者整体流式加载。相关的对照是论文引的 FlexGen 和 DeepSpeed-Inference 那种按层流式加载权重的做法，论文明确把它们归到另一条技术路线里，而且是为了吞吐型批量推理设计的。
3. **语义锚点这条对稠密模型部分还有意义，但收益来源不同。** 混合注意力架构（full attention 混 sliding-window attention 或循环层）的循环状态检查点锚定在特殊 token 边界这件事，跟 MoE 无关，稠密模型同样适用。但论文把这一条写在了「专家驻留」这个大框架里，且它**没有量化收益**，所以就算对稠密模型有用，论文也没给可迁移的数字。 [my inference]
4. **一条更微妙的限制：这个系统假设「激活参数能装进显存」。** 引言里举 DSV4-Flash 的例子时明确说 `At the deployed precision, this active parameter footprint fits within the 32GB memory capacity of an RTX 5090`。如果激活参数本身就装不进显存，前提就不成立了。这一点对稠密模型同样适用：稠密模型没有「激活参数小于全部参数」这个性质，所以它能被服务的规模上限就是显存上限。 [my inference]
5. **一处可能的推导边界问题。** 式 2 用 `max(B_H − B_P, 0)` 做了保护，但式 3 里 T_cpu 的分母直接写 `B_H − B_P`。如果 B_H < B_P，式 2 会把它钳到 0，式 3 就变成除以零或负数。所以这个模型只在 B_H > B_P 时成立。论文没有把这条写成显式前提，只在退化情形里讨论了「B_H 接近 B_P」。 [my inference]

---

## 6. 论文没有覆盖的东西（我自己的判断）

以下全部是 [my inference]，基于我读到的全文和代码。

1. **没有数据中心 / NVLink 的评测，一个数都没有。** 全部六台机器都是 PCIe 消费级或工作站级。所以「在数据中心 GPU 上会怎样」这个问题，论文既没有回答，也没有声明它不适用。
2. **没有质量或精度评测。** 论文声称合并是精确的、无算法近似，但**没有**任何精度或任务成功率的对照数字。文中只在描述 W2 和 W4 时提到「编码任务的运行必须产出参考 gold patch」「W4 必须完成全部十三轮」，算是一个通过/不通过的门槛，但**没有报通过率**。这算一个隐性正确性证据，不是一个评测。
3. **没有 q* 策略本身的实验。** 这是最大的一个缺口。论文花了 §3.2 一大段推导 q*，给了式 2、式 3、式 4，但**没有一个实验**比较「固定 q 或全部搬」与「q*」的延迟。Figure 2 只有一张示意。所以 q* 的收益在论文里是**分析性的，不是实测的**。
4. **没有语义锚点 KV 缓存的量化。** §3.1 花了不小篇幅讲这个机制，还引了 OpenClaw、OpenCode、SWE-agent 三个框架的具体编辑行为，但一个数字都没有。命中率、省下的预填充时间、消融，全部没有。
5. **没有 q* 对带宽测量误差的敏感性分析。** B_H 和 B_P 只在部署时测一次（代码里缓存成一张 per-GPU 的 JSON）。在真机上，用户的浏览器或游戏会实时改变可用带宽。论文强调弹性显存管理要处理这种波动，但**没有**讨论带宽波动对 q* 的影响，也没有说 B_H/B_P 是否需要在线重测。
6. **没有并发 / 批量维度的实验。** 所有解码数字都是「每请求平均 tok/s」。W3 提到 Claude Code 会派并发请求的子 agent，但论文没有报批量大小或并发度的影响，也没有说这套 CUDA Graph 加 CPU co-execution 在更大 batch 下是否还成立。CPU worker 池是绑核的持久池，大 batch 下的争用如何，没有数据。
7. **磁盘层没有被当作运行时层级来评测。** NVMe 只在启动成本里出现（读 140 GB 约 20 s）。论文的两层结构是「主机内存 + 显存」，磁盘只是加载来源。没看到「专家池大于主机内存时怎么办」的评测，虽然 GLM-5.2 那台机器上 KTransformers 恰恰是因为 512 GiB 装不下 753 GB 到 1.5 TB 而失败，说明这个场景是真实存在的。
8. **弹性显存重配只描述了机制，没有测。** §3.3 说可以在调度器安全点重建 GPU 专家缓存，不用重启引擎或重载主机专家池。但**没有**任何关于重建耗时、重建期间性能损失的实测。这对「用户边跑游戏边用」这个场景是关键数字。
9. **快速启动只有机制描述。** FTW（FreeToken Weight）格式、先读到最终布局再 pin 内存，这些做法都没有 FreeToken 自身启动时间的实测对比。只有一句「传统方式下 pin 空缓冲会导致 fault in 和清零若干 GB 页」的定性说明。
10. **图表是图片，不是数据。** 全文只有 Table 1 一张数值表。Figure 3、4、5 里的所有数字都只能从图注和正文叙述里读。这意味着我这份笔记里的大部分数字**无法被机器校验**，也没有附带原始日志或 artifact 链接。论文给了 GitHub 链接（`https://github.com/FlashML-org/FreeToken`，出现在 HTML 全文的 metadata 里）和 `https://flashml.ai`，但我没有去核对这两个 URL 是否存在，也没有去仓库里找实验脚本。
11. **20 个以上模型的支持是声明，不是评测。** 摘要和结论都说支持 20 个以上 MoE 模型，但 §5.1 明确只评了 3 个。中间的 17 个以上没有任何数据。
12. **没有功耗和能耗数字。** 对边缘设备来说这是一个重要维度（电池、散热），论文完全没有涉及。
13. **没有和「不 offload 的方案」对比。** 所有基线都是同样需要 offload 的边缘引擎。论文没有回答「如果显存够大，直接全放 GPU 会快多少」，所以看不出 FreeToken 离「显存无限」的上界有多远。
14. **代码与论文有一处对不上。** 论文说 q* 「always retains at least one fill」，我在 commit `cac247a` 的 `_ensure_experts_hybrid_kernel` 和 `_ensure_experts_hybrid_cpu` 里都没找到这条钳位。这条差异值得单独核对。 [my inference]

---

## 附：这份笔记里出现的所有 URL 及其抓取状态

| URL | 状态 | 用途 |
| --- | --- | --- |
| `https://arxiv.org/abs/2608.16157` | HTTP 200 成功 | 摘要、作者、日期、DOI |
| `https://arxiv.org/html/2608.16157v1` | HTTP 200 成功 | 全文正文、Table 1、所有图注 |
| `https://doi.org/10.48550/arXiv.2608.16157` | **没有抓取** | 这个 DOI 是从摘要页读到的，我没有访问它 |
| `https://github.com/FlashML-org/FreeToken` | **没有抓取** | 从全文 HTML 的 metadata 读到，没有核实 |
| `https://flashml.ai` | **没有抓取** | 同上 |

本地代码来源：`/tmp/FreeToken`，commit `cac247a`，`chore(release): 0.1.3 (#489)`。涉及的文件：`python/freetoken/moe/benchbw.py`、`python/freetoken/moe/bench_profile.py`、`python/freetoken/moe/offload_kernels.py`、`python/freetoken/moe/offload_cache.py`、`python/freetoken/engine/engine.py`、`tests/moe/test_hybrid_fetch.py`。

---

## 附：主控复核（由主控 agent 追加，未改动上文正文）

本节的目的是留下"这份笔记的关键结论被独立核对过"的痕迹。**我一条条抽查了下面四处，
全部通过**，并且补上了作者当时没有用上的一条独立证据。

**核 1：论文原句逐字对上。**
我把 `https://arxiv.org/html/2608.16157v1` 重新下载到本地（168785 字节），grep 明文：

```
always retains at least one fill, so the cache continues warming even when the
CPU handles most misses.
```

与笔记里引的那句完全一致。**通过。**
（说明：那一节还引了 `As B_H approaches B_P...` 与 q\* 闭式解，这两处在 HTML 里是
MathML，grep 抓不到明文，因此改用代码侧核对，见核 3、核 4。）

**核 2：取整规则不是 ceil 也不是 round。**
`python/freetoken/moe/offload_kernels.py:159-162`：

```python
lo = (m * frac_q16) >> 16
cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)
max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
num_fetch = min(len(missing), int(max_fetch))
```

确实是"在两个相邻整数里选让较慢那支更小的"，即 min-max。**通过。**
`tests/moe/test_hybrid_fetch.py:20-22` 的参考实现与注释也复述了同一条规则。

**核 3：代码首选竞争态实测值，回退才是 pcie/cpu，且与论文同源。**
`python/freetoken/moe/bench_profile.py:166-170` 的 docstring 写法完全对应：

```
Preferred source is the *overlapped* pair (CPU MoE and PCIe gather measured while
running concurrently -- the real contention regime): fetched/misses = pcie_ov /
(pcie_ov + cpu_ov). Older profiles without it fall back to the standalone bandwidths
under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA), which
reduces to pcie/cpu.
```

把 `cpu_ov = B_H − B_P`、`pcie_ov = B_P` 代入 `pcie_ov/(pcie_ov+cpu_ov)` 得
`B_P/B_H`，与论文闭式解一致。**通过。**
`tests/moe/test_hybrid_fetch.py:60-68` 用一个假 profile 验了这条优先级：
`bf16`（只有独立值）得 `0.4 = 40/100`，`nvfp4_x`（有竞争值）得 `0.25 = 30/(30+90)`，
若按独立值应为 `0.4`。所以"竞争态优先"是**有测试守着的**，不是注释里的空话。

**核 4：代码确实没有论文说的"至少保留一次 fill"，而且测试把它当作可接受行为。**
这一条原本是笔记里标记为"值得单独核对"的疑点，我核到底了，**结论是笔记对，代码与论文
不符**。两条独立证据：

1. 全仓 search 找不到任何 `max(1, ...)` 形式的钳位落在 fetch 数量上。
   `offload_kernels.py:162` 就是 `num_fetch = min(len(missing), int(max_fetch))`，
   `max_fetch` 可以算成 0。
2. **作者自己的单测主动断言了 f 可以为 0**。`tests/moe/test_hybrid_fetch.py:33-36`：

```python
for m in range(0, 65):
    f = _balanced_fetch(m, q)
    assert 0 <= f <= m          # 明确允许 f == 0
    assert abs(f - frac * m) <= 1.0
```

取 `m = 1`、`frac = 0.1` 即 `f = 0`，断言通过。也就是说这一步一个专家都不搬回
GPU 缓存，与论文"cache continues warming"的承诺相反。

**这一条要作为待办交给上游或作者核对**，不要当成我们的 bug：它是 FreeToken 自己
论文与实现的差异。如果 FastLLM 将来要抄这套 q\* 策略，**必须自己补上这个钳位**，
否则在缺失数很少、探测带宽比又小的时候会出现"长期一个都不补、缓存永不升温"的行为。
