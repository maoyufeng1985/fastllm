# TP4 all-reduce 微基准：1Cat pack32 在本机的结论

日期：2026-09-14
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP
目的：给 `docs/sm70_1cat_port_plan.md` 第 4 节第 1 项定论。先证明能赢再移植，
不赢就记录并跳过。

原型代码在 `/home/arproto/`（一次性丢弃物，不入库）。
原始输出：`/home/arproto/run1.txt`、`run2.txt`、`run3.txt`。

## 0. 结论

**不移植。** 1Cat 出厂配置（1 个 CTA、32 字节打包、它的 barrier）在 10 KiB 上
24.28 µs，比 NCCL 的 16.56 µs 慢 47%，是全部变体里最慢的一个。它的 barrier
单独测也比 FastLLM 现有的慢 36%。

根因是硬件。这台机器的 4 张 V100 **NVLink 全部 inactive**（
`nvidia-smi nvlink -s` 报 all links are inActive），卡间走 PCIe switch
（`nvidia-smi topo -m` 全是 PIX）。一次跨卡 barrier 在本机要 9.3 µs，而 1Cat
的每 token AR 预算 1.587 ms / 128 次 = 12.4 µs **低于本机单个 barrier 的成本**。
它的算法在这台机器上没有可用空间。

顺带量到一个可用的改动方向。图内独立 dest 上它赢过 NCCL（§4），但引擎调用仍是 in-place，还不能接线。

## 1. 变体与消息尺寸

decode 的 AR 消息是 hidden 5120 × 2 B = 10240 B（10 KiB），每 token 128 次
（64 层 × 2）。尺寸档取 10 / 20 / 40 / 80 KiB。

| 变体 | 几何 | barrier |
|---|---|---|
| nccl | ncclAllReduce FP16 | NCCL |
| ff-geom+ff-bar | 16 B 打包，grid = ceil(640/512) = 2 | FastLLM（release/acquire sys） |
| ff-geom+1cat-bar | 同上 | 1Cat（sys 发布 + device 域 volatile 轮询 + 一次 membar.sys） |
| pack32+1cat-bar | 32 B 打包，grid = 1 | 1Cat |
| 1block+16B+ff-bar | 16 B 打包，grid = 1 | FastLLM |
| barrier-only | 无 payload | 两种各一 |

`pack32+1cat-bar` 就是 1Cat 的
`sm70_cross_device_reduce_1stage_pack32<4><<<1, 512>>>`（
`csrc/custom_all_reduce.cuh:1984`）。它要求 `blocks == 1`，而
`defaultBlockLimit` 是 36，所以 1Cat 在验收里必须设
`VLLM_CUSTOM_ALLREDUCE_BLOCK_LIMIT=1` 才走得到。`1block+16B` 这一组就是把
「单 CTA」和「32 字节打包」两个因素拆开。

## 2. 实测（run3，128 次背靠背，取 4 卡最大值）

| 变体 | 10 KiB | 20 KiB | 40 KiB | 80 KiB |
|---|---:|---:|---:|---:|
| nccl | **16.56** | **19.29** | **20.99** | **30.69** |
| ff-geom+ff-bar（现役 shape） | 17.92 | 20.47 | 29.56 | 45.36 |
| ff-geom+1cat-bar | 21.69 | 24.15 | 33.23 | 49.33 |
| **pack32+1cat-bar（1Cat 出厂）** | **24.28** | 37.09 | 55.43 | 81.06 |
| 1block+16B+ff-bar | 18.37 | 24.36 | 32.43 | 48.54 |
| ff-geom+ff-bar(no end) | **14.94** | 17.53 | 26.22 | 42.57 |
| 1block+16B+ff-bar(no end) | 15.58 | 21.82 | 30.20 | 46.60 |
| pack32+ff-bar(no end) | 18.13 | 31.09 | 47.67 | 71.23 |
| barrier-only ff | 9.30 | 9.34 | 9.31 | 9.34 |
| barrier-only 1cat | 12.68 | 12.74 | 12.67 | 12.74 |

run1 是同一天早一轮的采样，排序完全一致，绝对值高 5% 到 10%（时钟状态）。

三次读法：

1. **同 barrier 比几何。** 1Cat 的 32 B / 单 CTA 在 10 KiB 是 18.13 µs，FastLLM
   的 16 B / 2 CTA 是 14.94 µs。几何本身就慢 21%，而且尺寸越大越差（80 KiB
   71.23 对 42.57）。单 CTA 只能用一个 SM 的未完成访存，喂不满 PCIe。
2. **同几何比 barrier。** 17.92 对 21.69 µs，1Cat 的 barrier 慢 21%。它的
   `membar.sys` 在轮询之后再加一次全系统栅栏，而 FastLLM 用的是
   `ld.acquire.sys`，每次轮询直接带序。
3. **barrier 是主要成本。** 10 KiB 的 payload 在 PCIe 上不到 1 µs，而
   FastLLM 的双 barrier 要 9.30 µs，占 17.92 µs 的 52%。加宽打包、减少
   CTA、换轮询方式都动不了这一项。

## 3. 现役 AR 与 NCCL

FastLLM 现役一阶段路径在 10 KiB 是 17.92 µs，NCCL 16.56 µs，现役慢 8%。
40 KiB 以上现役明显更差（29.56 对 20.99，45.36 对 30.69）。

这也解释了 trace 里 decode 每 token 的 128 次 `FastllmCustomAllReduceKernel`
共 2724.7 µs、平均 21.3 µs：本机单次集合通信的下限就在 9 到 16 µs 之间，
128 次就是 1.2 到 2.0 ms，占 decode 12.76 ms 的 9% 到 16%。

barrier 的次数而不是 payload 的大小决定这项成本。要往下走只有两条路，都不是
移植 1Cat 能拿到的：

- 减少每 token 的 barrier 次数（合并层间的 AR，或换成不需要完成确认的 push）；
- 用 NCCL 替掉 40 KiB 以上的现役路径。

落地（2026-09-14）：`FASTLLM_CUDA_CUSTOM_ALLREDUCE=auto` 的 small 探针改为
decode 真实尺寸 10 KiB；TP>=4 在 40 KiB 及以上无条件保留 NCCL，避免 10 KiB
偶尔过门把 40–512 KiB 也带上 custom。启动日志里 small 行现在报 `10 KiB`。

本机 in-graph 复测（`customAllReduceRegression`，4×V100，CUDA Graph replay）：

| 尺寸 | custom（force） | NCCL | 3% 门 | auto 实际路径 |
|---|---:|---:|---|---|
| 10 KiB | 18.31 µs | 20.03 µs | custom 略赢 | 探针噪声大，有时开有时关 |
| 20 KiB | 21.44 µs | 22.26 µs | custom 略赢 | 跟 small 路径走 |
| 40 KiB | 28.12 µs | 23.96 µs | **NCCL 赢** | 硬切 NCCL |
| 80 KiB | 42.33 µs | 33.81 µs | **NCCL 赢** | 硬切 NCCL |

40 KiB NCCL 图内 512 次回放三次均完成（22.5–27.5 µs/collective），无死锁。

## 4. 去掉结束 barrier：图内成立，引擎还接不上

eager 口径下，去掉结束 barrier 是全部变体里唯一赢过 NCCL 的：14.94 对 16.56 µs。
生产 decode 走 CUDA Graph，所以用 `/home/arproto/noend_graph_probe.cu` 在
per-thread stream、图回放、独立 dest 上重测（3 轮中位数，100 reps）：

| 模式 | 10 KiB | 16 KiB | 20 KiB | 40 KiB | 80 KiB |
|---|---:|---:|---:|---:|---:|
| graph-nccl-inplace | 16.05 | 16.51 | 18.33 | **20.54** | **31.37** |
| graph-nccl-ooo | 15.97 | 16.59 | 18.33 | 20.58 | 31.28 |
| graph-ff-end-ooo | 15.50 | 18.31 | 19.68 | 27.46 | 43.28 |
| graph-ff-noend-ooo | **12.94** | **15.56** | **16.84** | 25.72 | 41.95 |
| graph-ff-noend-chain3（每 AR） | 11.87 | 13.79 | 15.07 | 23.75 | 40.14 |
| graph-ff-noend-chain3-tailend（每 AR） | 12.69 | 14.62 | 15.97 | 24.52 | 40.70 |
| graph-nccl-inplace-chain3（每 AR） | 14.40 | 14.97 | 16.72 | 18.86 | 29.78 |

10 KiB 上单次 no-end 比 graph NCCL 快 19%（12.94 对 16.05）。3% 门过了。
原始表：`/home/arproto/noend_graph_run1.txt`。正确性：`/home/arproto/noend_graph_verify.txt`。

读表时注意三件事。

1. 头条用单次 12.94 对 16.05。链式行把一张 3 kernel 图的回放时间除以 3，摊掉了图启动开销，所以每 AR 比单次更快，不能单独当成"少一次 barrier 的收益"。
2. `--verify` 的 512 次回放只检查 4 卡结果一致，不检查绝对值。ping-pong 512 次会溢出 fp16，inf 对 inf 也会过。eager 单次和 chain3 有 `max_abs_err`。
3. 探针在循环里直接写 dest，对应 `fusedCopyBack=false`。引擎 decode 默认 `fusedCopyBack=true`，结果先放寄存器，结束 barrier 后再写。所以 `graph-ff-end-ooo` 不是今天引擎的基线。19% 是探针数字，不是 decode 墙钟。

40 KiB 以上 NCCL 仍然更快。auto 模式把 ≥40 KiB 硬切 NCCL 仍然对。

落地（2026-09-14）：one-stage custom 在 `data != dest` 时跳过 end barrier。
Qwen3.5 decode 热路径用 `Qwen3CudaTpReducePingPong`：独立 dest，交换 `cudaData`
而不是 `Data` 对象。CUDA Graph 在交替 launch 上烤进两套指针。in-place 路径
仍走 `writeAfterBarrier`。MTP 仍是 in-place。

21.3 µs/次已用 kernel 内计时拆开（barrier 5.1 + payload 6.1，其余是 end
barrier 和排队），TP4 push 变体在探针里快 2.2 倍。见
`docs/sm70_ar_microbench_deepdive.md` §6。

8K C=1 128 token，token sha256 与接线前一致
（`d5fcc5fca3d16030d35f6e0a5dab652e21bf0a74a67ec843419b139539fca2b0`）：

| | TPOP | decode |
|---|---:|---:|
| 接线前 | 13.31 ms | 75.13 tok/s |
| ping-pong skip-end | 13.16 ms | 76.02 tok/s |

省 0.15 ms/token，约 decode 的 1.1%。探针上 19% 是无排队的 AR 本身。引擎里
21.3 µs/次含排队，end barrier 只是其中一部分，所以墙钟远小于探针比例。
回归：`customAllReduceRegression` TP4 force-enable PASS，
`graph_pingpong_ordering=1`。

## 5. 复现方式

```sh
cd /home/arproto
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_70 \
    --default-stream=per-thread --expt-relaxed-constexpr \
    -Wno-deprecated-gpu-targets -o ar_bench ar_bench.cu -lcudart -lnccl -lpthread
./ar_bench --verify   # 36 项数值检查，全部 max_abs_err = 0
./ar_bench

/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_70 \
    --default-stream=per-thread --expt-relaxed-constexpr \
    -Wno-deprecated-gpu-targets -o noend_graph_probe noend_graph_probe.cu \
    -lcudart -lnccl -lpthread
./noend_graph_probe --verify
./noend_graph_probe --rounds=3
```

校验方式：harness 里 FastLLM 几何的重新实现落在 16 到 20 KiB 区间约 19 到
20 µs，与上一轮 in-graph 实测的 18.196 µs（16 KiB）在 8% 以内，与 trace 里
decode 的 21.3 µs 在 20% 以内。两个 1Cat 变体是照
`csrc/custom_all_reduce.cuh:491-620` 和 `:742-758` 逐句搬的，它们的落后不依赖
基线的绝对精度：同 barrier 比几何、同几何比 barrier 两次对照都指向同一结论。

## 6. 顺带：QPN2 过门验证

同一批原型里量了 GDN-in 的 pad 形状（`/home/arproto/qpn2_gdn.cu`）：

```
Nvfp4QpnCanRun(1,5120,4120)=0   (逻辑 GDN-in，被门挡住)
Nvfp4QpnCanRun(1,5120,4128)=1   (pad 后，过门)
```

| 形状（M=1） | standalone 图计时 | 有效带宽 |
|---|---:|---:|
| K5120 × N5120 | 23.76 µs | 620.6 GB/s |
| K5120 × N4128 | 22.70 µs | 523.6 GB/s |
| K5120 × N3584 | 22.14 µs | 466.1 GB/s |

standalone 的 N5120 是 23.76 µs，而同一 kernel 在引擎 trace 里是 17.04 µs，
说明 standalone 高估了约 1.4 倍，跨 harness 的绝对值不能直用。只用它做形状
缩放（N4128 / N5120 = 0.955），得到引擎内 N4128 约 16.3 µs。

对照引擎 trace 里 GDN-in 的 TurboMind 路径 28.2 µs 加 1.4 µs crop，每次省
约 12 µs，48 个 GDN 层共 **约 0.57 ms/token，decode 的 4.5%**。放宽到只用
standalone 数字则约 2%。取 2% 到 5%。

这一项的改动比 plan 预期的小：pad 已经在做（`kPackedOutputAlignment = 32`，
4120 → 4128），要改的是让 QPN2 侧车按 4128 建、输出复用同一个 crop。
