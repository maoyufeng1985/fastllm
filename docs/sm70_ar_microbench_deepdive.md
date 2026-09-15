# AR 成本归因深挖：barrier 次数 vs payload 大小

日期：2026-09-14
范围：Qwen3.8-27B-QUASAR-NVFP4 / SM70 / 4×V100-SXM2-16GB / TP4 / no-MTP
上游记录：`docs/sm70_ar_microbench.md`、`docs/sm70_1cat_port_plan.md` §4.1
原型：`/home/arproto/`（ar_bench.cu、nccl_wrap_probe.cu，本机可复跑）
本机 trace：`/tmp/nsys_dec.sqlite`（TensorRT/nsys 导出，decode-only workload）

---

## 0. 一句话结论

原判断的三条骨架上**两条成立、一条是错的**，而且错的那条正好是"往哪走"的依据：

| 原表述 | 判定 | 证据 |
|---|---|---|
| barrier 次数而不是 payload 大小决定成本 | **成立，且比原文更强** | 10 KiB payload 在 PCIe 上 <1 µs；barrier-only 9.28 µs |
| 每 token 128 次 AR，成本 ≈ 1.587 ms 预算 | **成立**，trace 实测到 4 个完整 128-AR epoch | epoch 9/10/11/12 恰好 128 个 |
| "现役路径"在 40 KiB 以上该被 NCCL 替掉 | **eager 成立、graph decode 不成立** | 见 §2 / §4：eager NCCL 全程快；graph 下 10–16 KiB 与 custom 打平 |
| 每 token AR 成本 = 128 × barrier = 1.2 ms | **错**，实测 2.6–8.8 ms | 见 §3 |
| 引擎 NCCL 慢是因为 `NcclSubmitRendezvous` | **错**，TP4 根本不建这个对象 | 见 §4：会合只在奇数 TP；TP4 税 ~1 µs |

原文说"barrier 次数决定成本"，然后据此推出"要么合并 AR，要么换 NCCL"。深挖之后：
**合并 / 去掉 end barrier 才是 decode（10 KiB、CUDA Graph）还没被证伪的方向。**
NCCL 那条路在 **eager** 下从 10 KiB 就赢，但生产 decode 走 CUDA Graph——graph 里 16 KiB NCCL ≈ 20 µs，与 custom 打平，auto-test 选 custom 是对的。
40 KiB 以上 NCCL 在 graph 里仍然明显更快，原文的"替掉大消息"在 **40–512 KiB** 这一档仍然成立。

---

## 1. 微基准已复跑，数字与记录一致

重跑 `./ar_bench`（同样的 4×V100，NVLink 全 inactive，全 PIX）：

| 变体 | 10 KiB | 20 KiB | 40 KiB | 80 KiB |
|---|---:|---:|---:|---:|
| nccl | **16.58** | **19.29** | **20.98** | **30.81** |
| ff-geom+ff-bar（现役 shape） | 17.97 | 20.80 | 29.77 | 45.14 |
| ff-geom+ff-bar(no end) | 15.00 | 17.62 | 26.02 | 42.30 |
| pack32+1cat-bar（1Cat 出厂） | 24.36 | 37.08 | 55.70 | 80.82 |
| barrier-only ff | 9.28 | 9.26 | 9.30 | 9.30 |
| barrier-only 1cat | 12.71 | 12.74 | 12.71 | 12.72 |

run3 是 16.56/17.92/24.28/9.30，本次 16.58/17.97/24.36/9.28——**复现误差 <0.3%**。
结论未变：1Cat pack32 最慢，barrier 是主要成本。

## 2. 修正一：NCCL 的优势起点不是 40 KiB，而是 10 KiB

原文只说"40 KiB 以上现役明显更差"，因此把换 NCCL 写成"覆盖大消息"的局部改动。
复跑数据（§1）把 ratio 摆平之后是：

| size | ff/nccl | 结论 |
|---|---:|---|
| 10 KiB | 1.084 | NCCL 快 8% |
| 20 KiB | 1.078 | NCCL 快 8% |
| 40 KiB | 1.419 | NCCL 快 42% |
| 80 KiB | 1.465 | NCCL 快 47% |

**eager 下 NCCL 在全部四档都快，10 KiB 也没有例外。** 但这是 `ar_bench` 的 eager
`ncclAllReduce`。生产 decode 默认 `cuda_graph=on`，必须看 graph 数字（§4）：
graph 16 KiB NCCL ≈ 20 µs，与引擎 auto-test 的 custom 19.95 µs 打平，
所以 decode 主力消息**不该**换成 NCCL。40 KiB 以上 graph NCCL 仍明显更快
（22.6 vs 29.8），原文"替掉大消息"只在这一档成立。

## 3. 修正二：每 token 的 AR 成本被低估了 2–7 倍

这是本次最重要的发现。原文写：

> 它的每 token AR 预算 1.587 ms / 128 次 = 12.4 µs **低于本机单个 barrier 的成本**

这个算式取的是 1Cat 的预算，不是 FastLLM 的实测。从 trace 里直接量 FastLLM 自己：

`/tmp/nsys_dec.sqlite`（decode-only：入 2048 / 出 32 / batch 1 / 14.04 ms/token）。
把 dev0 上所有 AR kernel 按 >20 ms 间隔切成 epoch，**有 4 个 epoch 恰好各 128 个 AR**，
这就是"每 token 128 次 AR"的直接证据（与 64 层 ×2 的推演一致）：

| epoch | span | AR 数 | AR 总时长 | 单次均值 | 占 14.04 ms |
|---|---:|---:|---:|---:|---:|
| 9 | 53.1 ms | 128 | 2.95 ms | 23.1 µs | 21% |
| 10 | 41.2 ms | 128 | 3.74 ms | 29.2 µs | 27% |
| 11 | 39.2 ms | 128 | 5.32 ms | 41.6 µs | 38% |
| 12 | 43.8 ms | 128 | 8.79 ms | 68.7 µs | **63%** |

两点必须说清楚：

1. **单次 AR 在引擎里是 ~20–69 µs，不是 harness 的 ~10–18 µs。** harness 的
   barrier-only 下限 9.28 µs 在引擎里达不到——因为引擎里的 AR 与相邻 kernel
   在流上串行排队，单次 AR 的"观测时长"含了排队与 PCIe 争用。
2. **成本随 token 位置漂移 3 倍**（23 µs → 69 µs）。这说明 AR 成本不只有常量
   barrier，还有随内存压力/占用变化的成分。任何"减半 AR 成本"的预估都该给区间，
   而不是单点。

所以原文"每 token AR 成本 ≈ 1.2 ms"是**下界估计**，真实区间 **2.6–8.8 ms**，
占 decode 的 **21%–63%**。这反而**加强**了"这一项是第一优先"的判断——
但也说明**上限不是 15%–19%，而是可能到 60%**。

## 4. 修正三：`NcclSubmitRendezvous` 不是 TP4 的税；差在 CUDA Graph

上一版把 27.99 vs 16.58 归到 host rendezvous。**代码否掉了这个推断**：

```
src/devices/multicuda/fastllm-multicuda.cu:2266
if (numGPUs % 2 != 0) {
    g_ncclSubmitRendezvous.reset(new fastllm::NcclSubmitRendezvous(numGPUs));
}
```

会合对象**只在奇数 TP 上构造**（RTX 5090 上 odd-TP 死锁的补丁）。TP4 上
`g_ncclSubmitRendezvous` 一直是 null，`Wait(Before/After)` 是空操作。
auto-test 里 `ScopedNcclForceSync` 也把 `FastllmNcclPostSyncEnabled` 关掉了，
所以 27.99 vs 16.58 **不是 rendezvous，也不是 post-sync**。

`/home/arproto/nccl_wrap_probe.cu` 把会合抄出来、在 TP4 上强制打开，
三轮干净采样的中位数（100 reps，max-over-ranks）：

| mode | 10 KiB | 16 KiB | 20 KiB | 40 KiB | 80 KiB |
|---|---:|---:|---:|---:|---:|
| eager-ooo（= ar_bench nccl） | 16.54 | 17.02 | 19.26 | 20.97 | 30.63 |
| eager-inplace | 16.49 | 17.09 | 19.22 | 21.07 | 30.54 |
| eager-ooo+rdv（强制会合） | 17.78 | 18.24 | 18.57 | 20.68 | 30.40 |
| graph-inplace-pts（对齐引擎：per-thread stream、in-place、graph replay） | 20.13 | 20.11 | 20.78 | 22.62 | 32.15 |

读法：

1. **in-place 几乎零成本**（eager-ooo 与 eager-inplace 差 <0.1 µs）。
2. **会合税在 decode 尺寸上是 +1.2 µs**（16 KiB 17.02 → 18.24），大消息被 NCCL
   本身盖住，甚至测出负值。就算 TP4 误开会合，也解释不了 69%。
3. **CUDA Graph 才是主差**：16 KiB eager 17.09 → graph-inplace-pts 20.11，+3.0 µs。
   引擎 auto-test 的 NCCL 27.99 仍比这个 graph 数字高 ~8 µs——剩余差未闭合，
   可能是 auto-test 夹在模型加载后的时钟/占用，或单节点 graph replay 触发了
   NCCL proxy 的额外开销。但方向已经够用：生产 decode 走 graph，NCCL 的可比
   数字是 ~20 µs，不是 harness 的 ~17 µs。

对照引擎 auto-test（同一次 decode 跑、graph replay、in-place）：

```
FP16 small 16 KiB: custom 19.949 us, NCCL 27.993 us  -> custom enabled
FP16 large 1024 KiB: custom 242.498 us, NCCL 180.742 us -> NCCL retained
```

custom 19.95 vs 本探针 graph NCCL 20.11——**打平**。auto-test 选 custom 对 10–16 KiB
decode 是对的。harness 里 NCCL 赢 8% 是 eager 口径，不能搬到 graph decode。

40 KiB 以上另一回事：graph-inplace-pts 22.62 vs ff-geom 29.77（40 KiB）、
32.15 vs 45.14（80 KiB）。TP4 的 two-stage 门在 512 KiB，所以 **40–512 KiB
现在走 custom one-stage，而 graph NCCL 已经更快**——这才是原文"替掉大消息"
真正还活着的区间。

## 5. 两条路的现状

### 路 1：减少每 token 的 barrier 次数

**可用杠杆比原文写得多。** 树里已经有现成的"合并 AR"机制：

- `FastllmCustomAllReducePairAdd*`（`fastllm-custom-allreduce.cu:489/554`）——
  一次 barrier 里 reduce+add 两个张量，就是"合并两次 AR"。
- `FastllmCustomAllReducePushAddKernel`（:793）——TP=2 的 push 变体，
  连"完成确认"都省掉，是原文说的"换成不需要完成确认的 push"。

但两者都卡在门限上：`CustomArUsePushAdd` 要求 `devices.size()==2`，
`PushAdd` 还有 `bytes <= 1 MiB`；`CustomArUseTwoStage` 在 TP4 要
`bytes >= 512 KiB` 才启用，而 decode 的 10 KiB 永远走 one-stage 且**保留 end barrier**。

**结论：TP4 的 decode 路径上，"合并"与"push"两个机制都够不到。**
merge 只有把两个**独立**的 AR 合到一个 barrier 里才成立；transformer 里相邻的
AR 是数据依赖的（attention-out 的 AR 必须在 MLP 之前），所以"合并层间 AR"在这条
模型栈上不合法，除非改图让两个 AR 的目的缓冲独立（原文 §4 说的那件事）。

这才是真正该做的实验，而不是"移植 1Cat"。

### 路 2：用 NCCL 替掉现役路径

eager 下从 10 KiB 就赢（§1），**graph decode 下 10–16 KiB 打平，auto-test 选
custom 是对的**（§4）。还活着的区间是 **40–512 KiB**：two-stage 门在 512 KiB，
这一档现在走 custom one-stage，graph NCCL 已经快 24%–29%。
会合不是 TP4 的税，不用为它改引擎。

## 6. 下一步（按证据强度排序）

1. **21.3 µs 已拆开：barrier 5.1 µs + payload 6.1 µs，其余是 end barrier 和排队。**
   `/home/arproto/ar_phase_probe.cu` 用 kernel 内 `%globaltimer` 把现役
   one-stage（skip-end 后）拆相。10 KiB、2 block × 512 thread、图回放、
   128 AR/token、50 µs filler 模拟层间计算：
   - start barrier 等待：各卡均匀 4.1–5.1 µs。**均匀说明这是协议 RTT，
     不是 skew 吸收**——skew 会让先到的卡等很久、最后到的卡等 0。
   - payload（3 peer 拉取 + 累加 + 写出）：6.14 µs。
   - kernel 合计 11.3 µs，entry-to-entry gap 63.5 µs − 50 filler = 每 AR
     13.5 µs；背靠背（filler=0）每 AR 14.0 µs。
   - `%globaltimer` 跨卡不同步（固定偏移 ~698 ms），跨卡 skew 直测不了。
     探针内 token 无 drift（gap 稳定），老 trace 的 23→69 µs 漂移来自引擎
     特有的 KV/attention 争用，不是 AR 协议。
2. **TP4 push 探针赢了 2.2 倍。** 同一探针的 push 变体（payload 即信号，
   无 start barrier，+0 空槽标记发 −0，双 phase 防回放踩踏）：
   背靠背每 AR 6.3 µs（pull 14.0），50 µs filler 每 AR 5.3 µs（pull 13.5），
   kernel 内合计 ~3.6 µs（out 2.05 + poll 1–2）。数值校验 max_abs_err=0。
   省的是两块：flag 交换（~5 µs）和远端读轮询改本地轮询。引擎预估区间
   **0.3–1.0 ms/token（decode 的 2%–8%）**，下限留给排队和 KV 争用。
3. **下一刀：把 push 变体做成 FastLLM 的 TP4 decode AR。——暂缓（2026-09-15）。**
   中心收益 +3%–4% 不值得当前投入，决策见
   `docs/sm70_tp4_push_ar_plan.md` 状态行。方案与探针保留，可随时执行。
4. **PairAdd 结构性不可行。** attn AR 的输出经 RMSNorm、gateup、swiglu、down
   才到 MLP AR 的输入，两次 AR 数据依赖，不可能共享一个 barrier
   （`qwen3_5.cpp:11780-11820`）。
5. **PushAdd 现状是 TP2 专属**（`CustomArUsePushAdd` 要求
   `devices.size() == 2`，`fastllm-custom-allreduce.cu:993`）。第 3 项就是
   把它的机制推广到 TP4。
6. 40–512 KiB 继续走 NCCL（图内 NCCL 快 24% 以上）。不要移植 1Cat pack32
   （每尺寸都最慢）。不要为 TP4 动 `NcclSubmitRendezvous`（TP4 不建）。

## 7. 仍未验证

- §3 的 4 个 128-AR epoch 来自 decode-only trace（入 2048 / 出 32）。C>1 与长上下文
  下 AR 的数量与单次成本都未测，`sm70_1cat_port_plan.md` §5 的空白仍然在。
- 引擎 auto-test 的 NCCL 27.99 µs 仍比本探针 graph-inplace-pts 20.11 µs 高 ~8 µs。
  会合与 post-sync 已排除；剩余差可能是时钟/占用，或 NCCL 单节点 graph replay
  的 proxy 开销。不影响"graph 下 16 KiB custom 与 NCCL 打平"的决策。
- 单次 AR 的 20–69 µs 含排队时间，无法从 trace 里分离出"纯 barrier"与"排队"。
  要分离得用 CUPTI 的 kernel-level 或改 kernel 内计时。
- epoch 12 的 68.7 µs 均值只占 128 中的 1 个 epoch，可能是该 token 位置的内存
  抖动而非稳态，需重复采样确认。
- fused-graph（一张图里连发多次 NCCL）探针把 communicator 打到 140+ µs，数字作废。
  本轮 `noend_graph_probe` 的 `graph-nccl-inplace-chain3` 是干净的（10 KiB 每 AR
  14.40 µs），可以替代那次作废测量。
- no-end 的墙钟收益还没在引擎 decode 上量过。12.94 vs 16.05 是探针数字。引擎里
  单次 AR 观测是 20–69 µs（含排队），改图后能省多少仍未知。
