#!/usr/bin/env python3
"""prefill_probe.py -- 量一次 prefill 的逐块墙钟 + 逐卡 util/mem 采样。

为什么不用现成的 `ftllm.cli benchmark`：它不打印逐块 `[Prompt] N Tokens` 行
（那行在 qwen3_5.cpp:23917，门是 `model->verbose`），而本任务要的是
"每块墙钟 + tokens/s"。这里复用 benchmark 的 _run_batch（同一套计时），
只额外打开 verbose 并在旁路按 0.5s 采样 nvidia-smi 的 sm%/mem%。

采样的 mem% 是显卡显存控制器的占用率（nvidia-smi utilization.memory），
用来做"计算受限 vs 带宽受限"的廉价判据：prefill 期间 mem% 接近 100% 而 sm%
不高 → 带宽受限；反之 → 计算受限。注意它是**占用率百分比**，不是 GB/s。

用法:
  python3 tools/prefill_probe.py --input_tokens 16384 --output_tokens 8 \
      --chunk_hint 2048 --out /tmp/pcab/i16_iso.out
"""
import argparse
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "build-sm70-tests", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fastllm_pytools"))

from ftllm.util import make_normal_parser, make_normal_llm_model  # noqa: E402
from ftllm.benchmark import (  # noqa: E402
    _build_input_tokens, _generation_args, _run_batch, _print_result,
)


def main():
    parser = make_normal_parser("prefill_probe")
    parser.add_argument("--input_tokens", type=int, default=16384)
    parser.add_argument("--output_tokens", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--prompt_unit", type=str,
                        default="FastLLM benchmark context block. ")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--repeat_penalty", type=float, default=None)
    parser.add_argument("--sample", type=float, default=0.5,
                        help="nvidia-smi 采样间隔秒")
    args = parser.parse_args()

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        samples.append((time.perf_counter(), int(parts[0]),
                                        int(parts[1]), int(parts[2])))
            except Exception:
                pass
            stop.wait(args.sample)

    model = make_normal_llm_model(args)
    try:
        model.set_verbose(True)
        gen = _generation_args(model, args)
        input_tokens = _build_input_tokens(model, args.input_tokens,
                                           args.prompt_unit)
        print("PROBE input_tokens=%d target_out=%d chunk_flag=%s" % (
            len(input_tokens), args.output_tokens,
            os.environ.get("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES",
                           "<unset>")), flush=True)
        for _ in range(args.warmup):
            _run_batch(model, input_tokens, min(args.output_tokens, 8), 1,
                       gen, label="warmup")

        t = threading.Thread(target=sampler, daemon=True)
        t.start()
        t0 = time.perf_counter()
        result = _run_batch(model, input_tokens, args.output_tokens,
                            args.batch, gen)
        wall = time.perf_counter() - t0
        stop.set()
        t.join(timeout=3)

        print("PROBE_RESULT wall=%.4fs ttft_avg=%s ttft_min=%s ttft_max=%s "
              "total_time=%.4f input=%d" % (
                  wall,
                  "%.4f" % result["ttft_avg"] if result["ttft_avg"] else "None",
                  "%.4f" % result["ttft_min"] if result["ttft_min"] else "None",
                  "%.4f" % result["ttft_max"] if result["ttft_max"] else "None",
                  result["total_time"], result["input_tokens"]), flush=True)
        print("PROBE_SHA %s" % result["token_hash"], flush=True)

        # 旁路采样汇总：只看引擎在忙的那段（wall 窗口内）
        if samples:
            start = min(s[0] for s in samples)
            busy = [s for s in samples if s[0] >= start]
            by_gpu = {}
            for _, idx, sm, mem in busy:
                by_gpu.setdefault(idx, []).append((sm, mem))
            for idx in sorted(by_gpu):
                vals = by_gpu[idx]
                sms = [v[0] for v in vals]
                mems = [v[1] for v in vals]
                print("PROBE_SAMPLE gpu=%d n=%d sm_avg=%.1f sm_max=%d "
                      "mem_avg=%.1f mem_max=%d" % (
                          idx, len(vals), sum(sms) / len(sms), max(sms),
                          sum(mems) / len(mems), max(mems)), flush=True)
        else:
            print("PROBE_SAMPLE none", flush=True)
    finally:
        try:
            model.release_memory()
        except Exception:
            pass


if __name__ == "__main__":
    main()
