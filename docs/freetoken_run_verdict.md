# FreeToken 在本机（4×V100 / SM70）跑不起来的判定

日期：2026-09-16
对象：`https://github.com/FlashML-org/FreeToken`，本地 shallow clone `/tmp/FreeToken`，
commit `cac247a`，`__version__ = 0.1.3`

## 0. 结论

**FreeToken 无法在这台机器上运行，因此 `FREETOKEN_HYBRID_OVERLAP=0/1` 的 A/B
在本机取不到数。** 这不是"没时间试"，是四个**独立的安装级硬门**加两条**硬件支持
声明**共同挡住的。下面每一条都能追到文件或命令。

本文件的作用是把这个判定钉死，并留下"在支持的机器上一条命令复现 A/B"的做法，
而不是留一句"没跑成"。

## 1. 安装级硬门（四条，任意一条即不可装）

| # | 门 | 本机现状 | 来源 |
|---|---|---|---|
| G1 | 驱动 **r580+（CUDA 13）** | **570.211.01**，`nvidia-smi` 自报最高 CUDA **12.8** | `docs/install.md` 的 Requirements：`driver r580+ (CUDA 13)`；本机 `nvidia-smi --query-gpu=driver_version` |
| G2 | 需要 **CUDA 13 toolkit + nvcc 在 PATH** | 只有 `/usr/local/cuda-12.8`，无 CUDA 13 | `docs/install.md`："CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH" |
| G3 | `torch>=2.11,<2.12`，且 uv 把它**钉死在 cu130 索引** | 系统无 torch；本机唯一现成 torch 是 1cat env 的 `2.10.0+cu128` | `pyproject.toml` `[build-system] requires`；`[tool.uv.sources] torch = { index = "pytorch-cu130" }`，索引 URL `https://download.pytorch.org/whl/cu130` |
| G4 | `accel` 额外依赖同为 CUDA 13 构建 | — | `pyproject.toml`：`fi = ["flashinfer-python[cu13]>=0.6,<0.7"]`、`sgl = ["sglang-kernel==0.4.5"]`（索引 `sglang-cu130`）、`accel = ["freetoken[fi,sgl]"]` |

补充一条实测：查询 PyPI `torch` 2.11.0 的 28 个发行文件，**没有** cp312 且
linux_x86_64 的 cu128 变体，所以也没有"退回 CUDA 12 的 torch"这条路。
（`pyproject.toml` 的注释本身也写了 PyPI 上 2.11.0 就是 cu130 构建。）

## 2. 硬件支持声明（两条）

* **README 列的支持面是本机没有的卡**："native support for NVIDIA RTX 30, RTX 40,
  and RTX 50 series GPUs"，即 SM86 / SM89 / SM120。本机是 V100，**SM70**。
* **NVFP4 专家 GEMM 路径明确排除 SM70**。
  `python/freetoken/layers/quantization/moe/nvfp4.py:241` 是
  `return (8, 0) <= backend.device_capability() < (10, 0)`，本机
  `device_capability() == (7, 0)`，不满足。
  `pyproject.toml` 的注释也写明 Marlin W4A16 NVFP4 专家 GEMM 是 `sm_80-99`。

**一条要说清楚的反例，免得把话说满**：并不是所有 kernel 都排 SM70。
`python/freetoken/kernel/csrc/jit/fast_index_copy.cuh:78` 是
`#if __CUDA_ARCH__ >= 700`，这道门 SM70 是过的。所以挡住的不是个别 kernel，
而是**整个运行时的装配**（torch cu130 + flashinfer cu13 + sglang-kernel cu130
都装不上）。

## 3. 因此 (c) 的处置

原计划是在本机跑：

```bash
# 原计划（本机不可执行）
FREETOKEN_HYBRID_OVERLAP=1 python benchmarks/bench_decode_moe.py --model <MODEL> --backend hybrid
FREETOKEN_HYBRID_OVERLAP=0 python benchmarks/bench_decode_moe.py --model <MODEL> --backend hybrid
```

**不在本机执行**，理由见 §1、§2。没有替代品能顶替它：A/B 要量的是
FreeToken 自己的 hybrid 路径，本机装不上 FreeToken，任何近似都只是另一件事。

**顺带把一个容易踩的坑记下来**：`FREETOKEN_HYBRID_OVERLAP` 是**模块级常量**，
在 import 时读一次（`python/freetoken/layers/moe.py:24-26`）。所以它必须在进程
启动前设进环境，**不能在进程内中途切换**；想在一个进程里跑两边是做不到的。

### 3.1 这个开关到底切了什么（读源码得到，供将来在支持的机器上复核）

`python/freetoken/layers/moe.py` 的 `_decode_hybrid`：

* 先把 CPU 侧任务发出去并**不等**：`pending = executor.decode_submit(...)`
  （`decode_submit` 自己只发 D2H 拷贝和 CPU 池任务就返回，
  见 `python/freetoken/moe/cpu_executor.py:543`）。
* 然后 `cache.copy_missing()` 与 GPU 侧 GEMM 照常跑。
* 关掉重叠时（`_HYBRID_OVERLAP == False`），在**发 GPU 工作之前**就把 CPU 同步掉
  （`layers/moe.py:325-328`：`cpu_routed_early = executor.decode_sync(pending) if
  not _HYBRID_OVERLAP else None`），于是 CPU 算与 GPU 取+算被串起来；
  开着时同步被推到 GPU GEMM 之后（`layers/moe.py:344`）。
  两边的差就是"CPU 计算被 GPU 取数+GEMM 藏掉了多少"。

所以这个 A/B 的口径是干净的，**它是 FreeToken 自己提供的重叠隔离开关**，
不是外部拼出来的对照。

## 4. 在支持的机器上要跑什么（一条命令的复现）

前置：驱动 r580+、CUDA 13 toolkit、一张 SM86/89/120 的卡、一个被支持的 MoE
checkpoint（README 举例 DeepSeek-V4-Flash、Qwen3.6-35B-A3B、GLM-5.2）。

```bash
git clone https://github.com/FlashML-org/FreeToken && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

# 先让引擎知道本机的带宽配比（决定 hybrid/offload 与 q* 切分）
ft bench bw

# A/B：只差一个环境变量
for M in 1 0; do
  FREETOKEN_HYBRID_OVERLAP=$M python benchmarks/bench_decode_moe.py \
    --model /path/to/model --backend hybrid 2>&1 | tee /tmp/ft_ov$M.log
done
```

要读三个东西，缺一不可：

1. **两边必须都真的走了 hybrid**。`ft bench bw` 的输出决定 `recommend()` 的结论
   （`python/freetoken/moe/benchbw.py:599`：CPU 带宽超过 PCIe 的 2 倍才给
   `hybrid`，否则 `offload`）。如果它判成 `offload`，那 `--backend hybrid` 也拿不到
   重叠，A/B 的差会是 0，那是**配置没到位**而不是重叠没用。
2. **tok/s**，同 prompt、同采样参数、多跑几次取中位。
3. **CPU 是否真在算**，即 hybrid 里 CPU 分担的专家数不为 0。

**门**：`FREETOKEN_HYBRID_OVERLAP=1` 相对 `=0` 有可复现的正收益，才算重叠成立；
若差值落在噪声内，就是重叠没赚到，与 FreeToken 自己的说法相反，那一份结论要重写。

## 5. 未取到的东西（明确列出）

* **本机没有 FreeToken 的任何实测数据。** 本文件的每一条都来自源码、文档或本机
  环境查询，没有一条是跑出来的。
* **没有跑 `bench_decode_moe.py`、`ft bench bw`、`bench_offload_cache_copy.py`**
  中的任何一个。
* **没有跑任何"近似替代"探针。** 曾考虑用合成探针量"拷贝引擎与 SM 的重叠"，
  决定不做：它量的不是 FreeToken，而且本会话早先已经因为把合成探针的数当引擎
  证据而撤过一次错，不再重复同类操作。
