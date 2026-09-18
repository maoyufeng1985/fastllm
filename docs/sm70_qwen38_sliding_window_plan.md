# Qwen3.8-27B 滑窗省 KV：现成源码、理论收益、质量验证计划（2026-09-18）

范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4。
本文回答三件事。滑窗的代码要不要去 GitHub 找，理论收益是多少，开窗之后怎么量质量。

**状态：全部未实测。** 本文的每个百分比都是**计算**，前提写在 §5。没有任何一行是跑出来的。

## 0. 结论

1. **滑窗代码不用去 GitHub 找，这个仓库里就有**，C++，吃同一套分页 KV，参考模型是 Laguna（`src/models/laguna.cpp`）。
2. **DeepSeek-V4 的压缩行加 indexer 套不到 Qwen3.8**。压缩行和挑 token 的分数都是训练出来的权重，Qwen3.8 的 checkpoint 里没有，config 里连对应的键都没有。
3. **理论收益的上界**：内存侧一条 200K 请求从"装不下"变成 20 条；算力侧 80K 上下文的 decode 每步 attention 占 26%，开 8192 的窗理论上省掉其中大部分，TPOP 从 14.91 掉到约 11.4 ms（−23%）。两条都是计算。
4. **质量只能测，不能保证**。三层卡：窗口内逐 token 相同、超窗后量退化曲线、堵住只在个别请求上出错的工程边界。

## 1. 现状数字（实测）

来源 `/home/nsys_base_80k.log`，Qwen3.8-27B NVFP4，TP4（cuda:0,1,2,3），CUDA Graph 开，FP16 KV，80K 上下文，batch 1。

AutoWarmup 段：

```
localKVPerPage=2.10 MB      tokenGrowingLayers=16      availForKV=1.20 GB
KV Cache Token limit: 167936 tokens (totalPages=1312, pageLen=128)
Batch limit: 1
```

汇总段：

```
Total time      42.9564 s
TTFT avg        41063.13 ms
TPOP avg        14.91 ms/token
Prefill         1994.98 tokens/s
Token stream sha256 41c7fab5e8c9b43b7604ef0bdc703d90b5950282f1273dacfbe114a0eb2c2581
```

推出来的两个量（计算）：

- 每卡每 token 的 KV = 2.10 MB ÷ 128 = **16.4 KB**。对得上 FP16、每卡 1 个 KV head、16 层（4 个 KV head 在 TP4 上每卡各 1 个）。
- `tokenGrowingLayers=16` 印证了只有 16 层会长 KV，另外 48 层是线性注意力（GDN）。

当前线上服务（实测，`ps` 读到 PID 712815，已跑 44 分钟）：

```
ftllm.server /home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto \
  --kv_cache_dtype fp8_e4m3 --low_gpu_mem --gpu_mem_ratio 0.98 \
  --prefix_cache true --chunked_prefill_size 8192 --max_batch 8 --port 8080
```

## 2. decode 每步的 attention 耗时（实测输入 + 计算）

命令：

```bash
nsys stats --report cuda_gpu_kern_sum --format csv /home/nsys/dec80k.sqlite
nsys stats --report cuda_gpu_kern_sum --format csv /home/nsys/dec180k.sqlite
```

| 轨迹 | 函数 | 次数 | 均值 | 单层一步 |
|---|---|---:|---:|---:|
| dec80k | `FastllmPagedAttentionSplitSm70GqaD256Kernel<half,half>` | 8576 | 144.0 µs | |
| dec80k | `FastllmPagedAttentionCombineGQAKernel<half,6>` | 8576 | 97.8 µs | **241.8 µs** |
| dec180k | `FastllmPagedAttentionSplitSm70GqaD256Kernel<half,half>` | 4480 | 286.6 µs | |
| dec180k | `FastllmPagedAttentionCombineGQAParallelKernel<half>` | 4480 | 18.2 µs | **304.8 µs** |

8576 ÷ 16 层 = 536 步（80K），4480 ÷ 16 = 280 步（180K）。所以这两个数就是"一层一步"。
上下文 80K 到 180K，单层 split 从 144 涨到 287 µs（×1.99 对 ×2.25），方向上支持"耗时随访问的 key 数走"。

16 层合计一步：

- 80K：241.8 µs × 16 = **3.87 ms**，占 TPOP 14.91 ms 的 **26%**（计算）。
- 180K：304.8 µs × 16 = **4.88 ms**（缺 180K 那次的 TPOP，占比没算）。

两次轨迹的 combine 函数不是同一个（97.8 µs 对 18.2 µs），说明中间代码变过。按 180K 那个 combine 算，
80K 那次的 16 层合计只有 2.6 ms，占 17%。**3.87 ms 这个数偏大。**

## 3. 树里现成的滑窗实现（不用抄外部代码）

| 位置 | 干什么 |
|---|---|
| `src/models/laguna.cpp:1048` `GPUForwardAttentionWindowLeft(layer)` | 每层开关。full 层返回 -1，滑窗层返回 `sliding_window - 1`。注释写明 FlashInfer 的 `window_left` 不含当前 token |
| `src/models/laguna.cpp:1056` `GPUForwardPagedCacheMaxPages(layer)` | 滑窗层给更小的页池，窗口外的页还回去。开 CUDA Graph 时各层页号空间必须一致，图模式下退回统一池 |
| `src/models/laguna.cpp:1173` | 快照和前前缀缓存按 `max(1, sliding_window - 1)` 只留窗口那么多 token |
| `src/models/laguna.cpp:1358-1368` | prefill 也按窗裁，`firstVisible = max(0, absoluteQuery - sliding_window + 1)` |
| `include/models/qwen3_cuda_common.h:1578` | `windowLeft` 是分页 attention 算子的 int 参数 |
| `src/devices/cuda/cudadevice.cpp:10768` | 从 intParams 取 `windowLeft`，默认 -1 |
| `src/devices/cuda/attention/fastllm-attention.cu:4088` | 传给 FlashInfer 的 `window_left`；`:3649-3714` 按页数截断 KV 长度 |
| `src/devices/cuda/attention/fastllm-attention.cu:3788, 3798` | 没有 FlashInfer 时断言 `windowLeft < 0`，报 "Sliding-window paged attention requires FlashInfer." |

**V100 上走不通这条链。** FlashInfer 在 CC 7.0 上被禁用（`/home/nsys_base_80k.log`：`FlashInfer attention disabled on GPU N (CC 7.0)`），
原生分页注意力也不支持滑窗（引自 `docs/fp4-kv-cache.md:21`）。

SM70 上能跑的原生滑窗代码有两份：

- `src/devices/cuda/attention/fastllm-attention.cu:1546` `FastllmCudaDFlashAttention(..., slidingWindow)`，草稿模型在用。
- `src/devices/cuda/models/dots3-note-kernels.cu:2262` `FastllmCudaDots3NoteSlidingAttentionPrefill`。

上游 `origin/master` 也有这些（`git grep -i sliding origin/master -- src include` 命中 20 个文件，含 gemma4、laguna、dots3_note、step3p5、qwen3_5）。

## 4. 外部现成源码（链接与许可都核对过）

只列实测存在的。每一条我都下载看过我引用的符号，**没有跑过任何一个**。

### 4.1 滑窗

| 仓库 | 文件 | 给你什么 | 许可 |
|---|---|---|---|
| ggml-org/llama.cpp | `src/llama-kv-cache-iswa.h` | 两份 KV cache 实例，一份给非滑窗层，一份给滑窗层 | MIT |
| vllm-project/vllm | `vllm/v1/core/single_type_kv_cache_manager.py:946` | `SlidingWindowManager`，窗口外的块可回收，`null block` 补齐（`:277`） | Apache-2.0 |
| vllm-project/vllm | `vllm/v1/attention/backends/flash_attn.py:161` | `window_size=(sliding_window-1, 0)` | Apache-2.0 |
| flashinfer-ai/flashinfer | `flashinfer/decode.py:318` | `window_left` 参数，本引擎已在调 | Apache-2.0 |
| Dao-AILab/flash-attention | `csrc/flash_attn/flash_api.cpp:155-160` | `window_size_left/right` 的掩码语义 | BSD |
| huggingface/transformers | `src/transformers/cache_utils.py:213` | `DynamicSlidingWindowLayer`，k/v 只留 `min(seq_len, sliding_window)` | Apache-2.0 |

### 4.2 压缩行 + indexer（要训练权重）

| 仓库 | 文件 | 给你什么 | 许可 |
|---|---|---|---|
| ggml-org/llama.cpp | `src/llama-kv-cache-dsv4.h` | `llama_dsv4_comp_state`，收 `ratio`/`state_size`/`n_embd_state`，每层存 kv 与 score | MIT |
| ggml-org/llama.cpp | `src/llama-kv-cache-dsa.h` | 一份 cache 存 key，一份专存 lightning indexer 的 key | MIT |
| flashinfer-ai/flashinfer | `flashinfer/mla/_sparse_mla_sm120.py` 等 | SM120 的 V4 稀疏 MLA 解码 | Apache-2.0 |
| deepseek-ai/DeepGEMM | `deep_gemm/__init__.py` | `fp8_mqa_logits`、`fp8_paged_mqa_logits`、`get_paged_sparse_mqa_logits_metadata` | MIT |
| sgl-project/sglang | `python/sglang/srt/layers/attention/nsa/` | `nsa_indexer.py` 等一整套 | Apache-2.0 |
| vllm-project/vllm | `vllm/v1/attention/backends/mla/indexer.py` | indexer 路径 | Apache-2.0 |
| fla-org/native-sparse-attention | `native_sparse_attention/modeling_nsa.py`、`ops/parallel.py` | NSA 论文自己的实现，三条路是压缩、选择、滑窗 | MIT |
| MoonshotAI/MoBA | `moba/moba_efficient.py`、`moba_naive.py` | 块级选择，带朴素对照 | MIT |
| THUDM/IndexCache | 仓库根目录 | 跨层复用 index | Apache-2.0 |

### 4.3 无训练的淘汰与池化（不改权重）

| 仓库 | 文件 | 干什么 | 许可 |
|---|---|---|---|
| mit-han-lab/streaming-llm | `streaming_llm/kv_cache.py`、`pos_shift/modify_llama.py` | sink 加滚动窗口，配位置平移 | MIT |
| FMInference/H2O | `h2o_hf/utils_hh/modify_llama.py` | 按累计注意力分数淘汰 | 根目录没有 LICENSE 文件（`LICENSE`/`.txt`/`.md` 全 404） |
| FasterDecoding/SnapKV | `snapkv/monkeypatch/llama_hijack_4_37.py` | 用观测窗口挑要留的 KV 位置 | Apache-2.0 |
| mit-han-lab/Quest | `quest/models/QuestAttention.py`、`quest/ops/csrc/` | 按页选 top-k，和分页 KV 形状最贴 | MIT |
| Zefan-Cai/KVCache-Factory | `pyramidkv/pyramidkv_utils.py` | PyramidKV 的新地址，按层分配预算 | MIT |
| NVIDIA/kvpress | `kvpress/presses/snapkv_press.py` | press 集合，接口统一 | Apache-2.0 |
| microsoft/MInference | `minference/patch.py` | prefill 阶段稀疏，不是 decode 淘汰 | MIT |

## 5. 为什么 V4 那套套不上

V4 的压缩行来自训练出来的压缩器，挑 token 的 indexer 也是训练出来的。加载时要读这些张量：

- `src/models/deepseekv4.cpp:3391-3395`：`.indexer.wq_b.weight`、`.indexer.weights_proj.weight`、`.indexer.compressor.wkv.weight`、`.indexer.compressor.wgate.weight`、`.indexer.compressor.ape`。
- 每层档位来自 config 的 `compress_ratios`（`include/models/deepseekv4.h:11-13`、`:439-440`）：0 纯滑窗，4 是 CSA 加 indexer，128 是 HCA。
- 缓存结构：`windowKV` FP32 `[bsz, 128, 512]` 环形；`compressedKV` BF16 `[bsz, blocks, 512]`，`blocks = 已生成长度 / compress_ratio`（`deepseekv4.cpp:1384`）；`indexerCompressedKV` BF16 `[bsz, blocks, 128]`，只 ratio=4 的层有。

Qwen3.8-27B 没有这些：`/home/models/Qwen3.8-27B-QUASAR-NVFP4/config.json` 的 `text_config` 里没有任何 indexer 或 compress 键，
`src/models/qwen3_5.cpp` 里 grep 不到 indexer 或 sparse 代码。

同一仓库里的先例是 Qwen4-Exp 的 QSA：它从 config 读 `indexer_n_heads`、`indexer_head_dim`、`indexer_budget`、`indexer_compress_ratio`
（`src/models/qwen4_exp.cpp:2216-2220`），并加载 `indexer.index_qk_proj.weight` 等（`:2102`、`:2727-2729`）。
**代码支持这个形态，前提是 checkpoint 自带那批权重。**

## 6. 理论收益（两张表，全是计算）

**转移检查三行：**

1. `原测量对象` = Qwen3.8-27B NVFP4 / 4×V100 / TP4 / FP16 KV，80K 与 180K 上下文的 decode
2. `现结论的对象` = 同一个模型、同一台机器、同样 4 卡，只是那 16 层加了滑窗
3. `两者关系` = 同一个对象，差别只有加不加窗口，所以 §2 的单层耗时可以直接当"没加窗时"的输入

### 表一，KV 驻留（内存侧，每卡）

| 项 | 现在 | 开窗 8192 后 | 变化 | 状态 |
|---|---:|---:|---:|---|
| 一条 200K 请求占的 token 槽位 | 200000（超过上限 167936，装不下） | 8192 | −96% | 计算 |
| 池子能同时装几条 200K 请求 | 0 | 约 20 条 | — | 计算，仅内存侧 |
| 每 token 每卡 KV 字节 | 16.4 KB | 16.4 KB | 不变 | 由实测 `localKVPerPage=2.10 MB` 除页长 128 得到 |

### 表二，decode 每一步的时间（算力侧，80K 上下文）

| 项 | 现在（实测输入 + 计算） | 开窗 8192 后（计算） |
|---|---:|---:|
| 16 层 attention 一步合计 | 3.87 ms | 约 0.39 ms |
| 该步 TPOP | 14.91 ms | 约 11.43 ms |
| 相对变化 | — | **约 −23%** |

### 算这些数用到的假设，逐条列出

1. 注意力耗时与访问的 key 数成正比。decode 是一个 query 对全部 key，标准实现成立。**推断，未验。**
2. 两次轨迹的 combine 函数不同，80K 那次的 3.87 ms 偏大；按 180K 的 combine 算只有 2.6 ms（占 17%）。
3. 那些函数在 4 张卡上并行跑，单卡耗时约等于墙钟里这一段，前提是不与别的算子重叠。轨迹是 CUDA Graph 跑的。**未验。**
4. "池子能装 20 条"只管内存。日志里 `Batch limit: 1`，并发还受别的限制。
5. 窗口内换代码路径会改浮点累加顺序，可能改输出。§7 第一层就是查这个。

## 7. 质量验证计划

### 第一层，窗口内必须逐 token 相同

判据：上下文不超过窗口时，开窗和不加窗的输出 sha256 必须一致。

- 用贪婪解码，`--temperature 0`。
- 工具现成：`tools/fastllm_pytools/benchmark.py:458` 打印 `Token stream sha256`；`_token_stream_hash` 的注释（`:16`）写的就是"两次只差执行细节的跑法必须产生同一个贪婪流，比摘要即证明"。
- 边界值覆盖：正好等于窗口、窗口加一、窗口加一页（128）、窗口不是 128 整数倍。

这一层不过，后面的数没有意义。

### 第二层，超窗之后的退化，标准先定再跑

1. 先量不开窗的原模型在长上下文任务上的得分，那是唯一的对照。
2. 再定允许掉多少。这个数由需求方定。
3. 然后才跑开窗那次。

长上下文检索测试**仓库里没有**。`test/evaluation_plan/benchmark_catalog.json` 自己写着 `status: planning_only`，
里面的 `ruler`、`long_context` 是候选清单，不是能跑的测试。要自己写：把一句事实埋在 20 万 token 的不同深度（10%、50%、90%），问模型那句事实，看命中率。

现成能跑的下游测试是 `test/cmmlu`、`test/gsm8k`、`test/mmlu_pro`（走 API 的 eval 脚本，短上下文），
用来确认没把不相关的能力带崩。

两个扫描：逐层开窗（1 到 16 层，看第几层开始陡降，允许只开一部分层）、窗口大小（2048/4096/8192/16384，画命中率与并发条数的曲线）。

### 第三层，容易漏的边界

| 边界 | 要做的事 | 漏了会怎样 |
|---|---|---|
| attention sink | 最前面若干 token 永不裁 | 不留 sink 明显掉质量（引自 StreamingLLM，未在本引擎上验） |
| 位置编码 | 只改看哪些 KV，不改 position id | 位置信息错乱 |
| 48 层线性注意力 | recurrent state 不跟着裁 | 它是固定大小状态，不是 KV |
| 前缀缓存与快照 | 快照只留 `sliding_window - 1`（照 `laguna.cpp:1173`） | 恢复出的 KV 缺块，只在复用前缀的请求上错 |
| 分块 prefill | 块边界的窗口裁剪单独测（照 `laguna.cpp:1358-1368`） | 首块正常，后续块错位 |
| 投机解码与 MTP | 验证步窗口边界与主模型一致 | draft 与 verify 看的范围不同 |
| `--kv_cache_dtype fp4` 叠加 | 两层有损单独量组合 | 没人量过叠加结果 |
| CUDA Graph | 窗口边界变了要重新捕获（坑记在 `laguna.cpp:1056-1070`） | 重放读到错页 |

### 最小次序

1. 抓不开窗那次在窗口内上下文的贪婪 sha256，当基准。
2. 写 needle 测试，量原模型在 20K / 80K / 200K 上的命中率。
3. 实现窗口，跑第一层等价性，必须 100% 相同。
4. 跑 needle，对比第 2 步。
5. 组合回归：前缀缓存命中、并发、投机解码各跑一遍等价性。

## 8. 没做的部分

- 窗口一行都没实现。
- §2 的 attention 耗时来自 9 月 15 日的轨迹，中间代码变过，没在当天的 build 上复测。
- 180K 那次没有对应的 TPOP 日志，占比算不出来。
- 长上下文检索测试没写。
- 质量退化一个数都没有。

**推断（未验证）**：真正不掉质量的路只有让模型在训练时见到窗口化的注意力，V4 的压缩器和 indexer 就是这么来的。
推理引擎层面加窗口，是在拿质量换显存和速度。Qwen3.8 有 48 层线性注意力，它们的状态是整个前缀的固定大小摘要，
所以丢远处 KV 的损失可能比纯注意力模型小。**这条我没证据，要验就得跑长上下文检索题。**
