#!/usr/bin/env python3
"""nsys_gap.py -- 从一个 nsys sqlite 里，把"prefill 窗口"切出来，量化 gpu0 上的
空闲（gap）并归因。

切窗口的判据（不靠时间轴直觉，按结构特征）：
  1. 只看 CUPTI_ACTIVITY_KIND_KERNEL（内核自身时长，不含 cudaStreamSynchronize 排空）。
  2. 找到 per-device 内核占用最高的一段连续区间（prefill 是长内核段）。
  3. 用窗口内出现的集合通信内核（名字含 nccl/allreduce/all_reduce/AR）与
     GEMM/反量化内核作为"这是 prefill 而非空载"的结构证据，印出来核对。

gap 归因（每条 gap 都问"这段时间主机在干什么"）：
  - 对每个 gap，查 CUPTI_ACTIVITY_KIND_RUNTIME（CUDA runtime API 调用）：
    有没有某个 runtime 调用横跨这个 gap（例如 cudaStreamSynchronize / cudaMemcpy / cudaMalloc）。
  - 分类：host_sync / host_memcpy / host_malloc / host_other / no_runtime_call（纯流停顿）。

用法: python3 tools/nsys_gap.py <sqlite> [--device 0] [--min-gap-us 50]
"""
import argparse
import sqlite3
from collections import defaultdict

AR_PAT = ("nccl", "allreduce", "all_reduce", "allgather", "all_gather",
          "reducescatter", "reduce_scatter", "push", "pull", "broadcast")
GEMM_PAT = ("gemm", "cutlass", "nvfp4", "wmma", "mma", "cublas", "sgemm",
            "hgemm", "attention", "flash", "softmax", "dequant", "quant")


def classify(name):
    n = (name or "").lower()
    if any(p in n for p in AR_PAT):
        return "AR"
    if any(p in n for p in GEMM_PAT):
        return "compute"
    return "other"


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--min-gap-us", type=float, default=50.0)
    ap.add_argument("--win-start", type=float, default=None,
                    help="窗口起点 ns（给定时不再自动找）")
    ap.add_argument("--win-end", type=float, default=None)
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.sqlite, uri=True)
    dev = args.device

    rows = con.execute(
        "select k.start,k.end,coalesce(dn.value,sn.value,'') as nm "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "left join StringIds dn on dn.id=k.demangledName "
        "left join StringIds sn on sn.id=k.shortName "
        "where k.deviceId=? order by k.start", (dev,)).fetchall()
    if not rows:
        print("NO KERNELS device=%d" % dev)
        return
    if args.win_start is not None and args.win_end is not None:
        w0, w1 = args.win_start, args.win_end
    else:
        # 自动：模型加载是零散加载内核 + 长间隙；prefill 是一整段无长间隙的密集内核。
        # 判据：从最后一个 >2s 的间隙之后开始，到最后一个内核结束。
        iv = merge([[s, e] for s, e, _ in rows])
        big = [i for i in range(1, len(iv)) if iv[i][0] - iv[i - 1][1] > 2e9]
        print("BIG_GAPS(>2s) n=%d at=%s"
              % (len(big), [round((iv[i][0] - iv[i - 1][1]) / 1e9, 2) for i in big]))
        if big:
            w0 = iv[big[-1] + 1][0]
        else:
            w0 = iv[0][0]
        w1 = iv[-1][1]

    inwin = [(s, e, nm) for s, e, nm in rows if s >= w0 and e <= w1]
    print("WIN device=%d dur=%.4fs n_kernels=%d"
          % (dev, (w1 - w0) / 1e9, len(inwin)))
    print("WIN_ABS start_ns=%d end_ns=%d" % (w0, w1))

    # 分类合计（内核自身时长，会重叠的按累加，占比用 busy union 做分母另报）
    cls = defaultdict(lambda: [0.0, 0])
    for s, e, nm in inwin:
        c = classify(nm)
        cls[c][0] += (e - s) / 1e9
        cls[c][1] += 1
    total = sum(v[0] for v in cls.values())
    print("--- kernel-time by class (sum of kernel self time) ---")
    for c in sorted(cls, key=lambda k: -cls[k][0]):
        d, n = cls[c]
        print("CLASS %-8s sum=%.4fs pct=%.1f%% n=%d" % (c, d, 100.0 * d / total, n))

    # busy union + gaps
    bu = merge([[s, e] for s, e, _ in inwin])
    busy = sum(e - s for s, e in bu) / 1e9
    win = (w1 - w0) / 1e9
    print("BUSY union=%.4fs (%.1f%% of window)  IDLE=%.4fs (%.1f%%)"
          % (busy, 100 * busy / win, win - busy, 100 * (win - busy) / win))

    # runtime calls (host side) for attribution
    rt = con.execute(
        "select r.start,r.end,n.name,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds n on n.id=r.nameId where r.start>=? and r.end<=? order by r.start",
        (w0, w1)).fetchall()
    # 也允许横跨窗口边界
    rt_wide = con.execute(
        "select r.start,r.end,n.name,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds n on n.id=r.nameId where r.end>? and r.start<? order by r.start",
        (w0, w1)).fetchall()
    mc = con.execute(
        "select m.start,m.end,m.bytes,m.copyKind from CUPTI_ACTIVITY_KIND_MEMCPY m "
        "where m.deviceId=? and m.end>? and m.start<? order by m.start",
        (dev, w0, w1)).fetchall()

    gaps = []
    for i in range(1, len(bu)):
        gs, ge = bu[i - 1][1], bu[i][0]
        if (ge - gs) / 1e3 >= args.min_gap_us:
            gaps.append((gs, ge))
    print("--- gaps >= %.1f us (n=%d) ---" % (args.min_gap_us, len(gaps)))
    buckets = defaultdict(lambda: [0.0, 0])
    for gs, ge in gaps:
        d = (ge - gs) / 1e9
        # 覆盖这个 gap 的 runtime 调用（取与 gap 重叠最多者）
        cover = []
        for rs, re_, nm, tid in rt_wide:
            ov = min(re_, ge) - max(rs, gs)
            if ov > 0:
                cover.append((ov, nm, tid, rs, re_))
        # 覆盖这个 gap 的 memcpy（host 侧）
        mcov = []
        for ms, me, by, ck in mc:
            ov = min(me, ge) - max(ms, gs)
            if ov > 0:
                mcov.append((ov, by, ck))
        if cover:
            cover.sort(reverse=True)
            ov, nm, tid, rs, re_ = cover[0]
            frac = 100.0 * ov / (ge - gs)
            key = "host_" + _rtshort(nm)
            buckets[key][0] += d
            buckets[key][1] += 1
            tag = "%s (covers %.0f%%, tid=%d)" % (nm, frac, tid)
        else:
            key = "no_runtime_call"
            buckets[key][0] += d
            buckets[key][1] += 1
            tag = "(no runtime call overlaps)"
        print("GAP dur=%.4fs %s -> %s : %s%s" % (
            d, _t(gs, w0), _t(ge, w0), tag,
            "  [memcpy: %s]" % mcov[0][1:] if mcov else ""))
    print("--- gap time by cause ---")
    for k in sorted(buckets, key=lambda x: -buckets[x][0]):
        d, n = buckets[k]
        print("CAUSE %-28s sum=%.4fs n=%d (%.1f%% of idle)"
              % (k, d, n, 100.0 * d / max(1e-9, win - busy)))


def _t(ns, w0):
    return "+%.4fs" % ((ns - w0) / 1e9)


def _rtshort(nm):
    n = nm or ""
    for key, short in (("cudaStreamSynchronize", "sync"),
                       ("cuStreamSynchronize", "sync"),
                       ("cudaMemcpy", "memcpy"), ("cudaMalloc", "malloc"),
                       ("cudaLaunch", "launch"), ("cudaFree", "free"),
                       ("cudaEvent", "event"), ("cudaStreamWaitEvent", "wait"),
                       ("cudaDeviceSynchronize", "dstor"), ("cudaHost", "host"),
                       ("cudaGraph", "graph")):
        if key in n:
            return short
    return "other"


if __name__ == "__main__":
    main()
