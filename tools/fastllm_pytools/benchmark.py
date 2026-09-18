import argparse
import ctypes
import os
import hashlib
import statistics
import time
from typing import Dict, List, Optional

from .util import make_normal_llm_model, make_normal_parser


def _token_stream_hash(requests: List[Dict[str, object]]) -> str:
    """Hash the per-request generated token ids.

    Two runs that differ only in an execution detail (CUDA graph on/off, for
    example) must produce the same greedy stream; comparing this digest is how
    a caller proves that without dumping every token.
    """
    digest = hashlib.sha256()
    for item in requests:
        digest.update(",".join(str(t) for t in item.get("token_ids", [])).encode())
        digest.update(b"|")
    return digest.hexdigest()


def _common_decode_window(requests: List[Dict[str, object]],
                          batch_end: float):
    """Span in which every request has finished prefill and is decoding.

    The start is the *last* first-token time, not the first one. Starting at
    the earliest TTFT charges a peer's still-running prefill to decode, which
    for serialized long-context prefills (C=2 at 80K) misreports the steady
    rate by an order of magnitude.

    Each request's first token is its prefill result, so it is never counted
    as a decode token. Returns (start, span, tokens); start is None when no
    request produced a token.
    """
    starts = [item["first_token_time"] for item in requests
              if item["first_token_time"] is not None]
    if not starts:
        return None, 0.0, 0
    start = max(starts)
    span = max(0.0, batch_end - start)
    tokens = sum(
        1
        for item in requests
        for index, stamp in enumerate(item["token_times"])
        if index > 0 and stamp > start
    )
    return start, span, tokens


def _decode_tokens_before_last_ttft(requests: List[Dict[str, object]],
                                    last_ttft: Optional[float]) -> List[int]:
    """Generated tokens (not the first) stamped before the last request's TTFT.

    For C=2 this is the plan's yield gate: request #0 must keep decoding while
    #1 is still prefilling. Common-window rates start at last TTFT, so they
    cannot see that window.
    """
    if last_ttft is None:
        return [0 for _ in requests]
    counts = []
    for item in requests:
        counts.append(sum(
            1
            for index, stamp in enumerate(item["token_times"])
            if index > 0 and stamp < last_ttft
        ))
    return counts


def _per_request_common_rate(item: Dict[str, object], start: Optional[float]) -> float:
    """Decode rate for one request measured inside the common window."""
    if start is None:
        return 0.0
    own_end = item["end_time"]
    if own_end is None:
        return 0.0
    span = max(0.0, own_end - start)
    if span <= 0:
        return 0.0
    tokens = sum(
        1
        for index, stamp in enumerate(item["token_times"])
        if index > 0 and stamp > start
    )
    return tokens / span


def add_benchmark_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input_tokens", type=int, default=64,
                        help="Input token length for benchmark prompts")
    parser.add_argument("--output_tokens", type=int, default=256,
                        help="Max output token length for each benchmark request")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of concurrent benchmark requests")
    parser.add_argument("--stagger_s", type=float, default=0.0,
                        help="Delay each request's launch by this many seconds "
                             "relative to the previous one (0 = launch all back to back)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup requests before benchmark")
    parser.add_argument("--prompt_unit", type=str,
                        default="FastLLM benchmark context block. ",
                        help="Text unit repeated to build the synthetic prompt")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature; set <= 0 for greedy decoding")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Generation top_p")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Generation top_k")
    parser.add_argument("--repeat_penalty", "--repetition_penalty",
                        dest="repeat_penalty", type=float, default=None,
                        help="Generation repetition penalty")


def args_parser():
    parser = make_normal_parser("fastllm_benchmark")
    add_benchmark_args(parser)
    return parser.parse_args()


def _validate_args(args):
    if args.input_tokens <= 0:
        raise ValueError("--input_tokens must be greater than 0")
    if args.output_tokens <= 0:
        raise ValueError("--output_tokens must be greater than 0")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to 0")
    if args.prompt_unit == "":
        raise ValueError("--prompt_unit must not be empty")


def _encode_prompt(model, prompt: str) -> List[int]:
    if getattr(model, "hf_tokenizer", None) is not None:
        from .llm import encode_hf_prompt

        return encode_hf_prompt(model.hf_tokenizer, prompt)
    return model.encode(prompt)


def _build_input_tokens(model, target_tokens: int, prompt_unit: str) -> List[int]:
    repeat = 1
    token_ids = _encode_prompt(model, prompt_unit)
    if len(token_ids) == 0:
        raise ValueError("prompt_unit produced empty tokens")
    while len(token_ids) < target_tokens:
        repeat = max(repeat + 1,
                     int(repeat * target_tokens / max(len(token_ids), 1)) + 1)
        token_ids = _encode_prompt(model, prompt_unit * repeat)
    return token_ids[:target_tokens]


def _generation_args(model, args) -> Dict[str, object]:
    default_config = getattr(model, "default_generation_config", {})
    top_p = args.top_p if args.top_p is not None else default_config.get("top_p", 0.8)
    top_k = args.top_k if args.top_k is not None else default_config.get("top_k", 1)
    temperature = (args.temperature if args.temperature is not None
                   else default_config.get("temperature", 1.0))
    repeat_penalty = (args.repeat_penalty if args.repeat_penalty is not None
                      else default_config.get("repetition_penalty", 1.0))

    do_sample = True
    if temperature is not None and temperature <= 0:
        do_sample = False
        temperature = 1.0
        top_p = 1.0
        top_k = 1

    return {
        "do_sample": do_sample,
        "top_p": float(top_p),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "repeat_penalty": float(repeat_penalty),
    }


def _launch_raw_response(model, input_tokens: List[int], output_tokens: int,
                         generation_args: Dict[str, object]) -> int:
    from .llm import fastllm_lib

    stop_token_len, stop_token_list = model.stop_token_ctypes(None)
    input_buffer = (ctypes.c_int * len(input_tokens))(*input_tokens)
    return fastllm_lib.launch_response_llm_model(
        model.model,
        len(input_tokens),
        input_buffer,
        ctypes.c_int(output_tokens),
        ctypes.c_int(0),
        ctypes.c_bool(generation_args["do_sample"]),
        ctypes.c_float(generation_args["top_p"]),
        ctypes.c_int(generation_args["top_k"]),
        ctypes.c_float(generation_args["temperature"]),
        ctypes.c_float(generation_args["repeat_penalty"]),
        ctypes.c_bool(False),
        stop_token_len,
        stop_token_list,
    )


def _run_batch(model, input_tokens: List[int], output_tokens: int,
               batch: int, generation_args: Dict[str, object],
               label: str = "benchmark",
               stagger_s: float = 0.0) -> Dict[str, object]:
    from .llm import fastllm_lib

    requests = []
    batch_start = time.perf_counter()
    for request_id in range(batch):
        # Stagger mode: request N is launched stagger_s after request N-1. The
        # wait happens BEFORE this request's own start_time is taken, so the
        # reported TTFT for each request is still measured from its own launch.
        if request_id > 0 and stagger_s > 0:
            time.sleep(stagger_s)
        start_time = time.perf_counter()
        handle = _launch_raw_response(model, input_tokens, output_tokens, generation_args)
        requests.append({
            "request_id": request_id,
            "handle": handle,
            "start_time": start_time,
            "first_token_time": None,
            "end_time": None,
            "output_tokens": 0,
            "finish_code": None,
            "token_ids": [],
            "token_times": [],
        })

    pending = set(range(batch))
    while pending:
        progressed = False
        for request_index in list(pending):
            item = requests[request_index]
            if not fastllm_lib.can_fetch_response_llm_model(model.model, item["handle"]):
                continue
            token = fastllm_lib.fetch_response_llm_model(model.model, item["handle"])
            now = time.perf_counter()
            progressed = True
            if token <= -1:
                item["end_time"] = now
                item["finish_code"] = token
                pending.remove(request_index)
                continue
            if item["first_token_time"] is None:
                item["first_token_time"] = now
            item["output_tokens"] += 1
            item["token_ids"].append(int(token))
            item["token_times"].append(now)
        if not progressed:
            time.sleep(0.0005)

    batch_end = max(item["end_time"] for item in requests)
    # 定位用：把每条请求的绝对时刻（相对本次批量开始）打出来。判断"错开有没有
    # 真的发生""是不是所有请求同时在跑"，看这张表就够，不用再推。
    # 默认关闭；FASTLLM_BENCH_TIME_TRACE=1 打开。
    if os.environ.get("FASTLLM_BENCH_TIME_TRACE", "0") not in ("", "0"):
        for it in requests:
            ftt = it["first_token_time"]
            print("[reqtime] id=%d launch=%.3fs ttft=%s first_token@%.3fs end=%.3fs "
                  "tokens=%d decode_span=%.3fs" % (
                it["request_id"],
                it["start_time"] - batch_start,
                ("%.3fs" % (ftt - it["start_time"])) if ftt else "None",
                (ftt - batch_start) if ftt else -1.0,
                it["end_time"] - batch_start,
                it["output_tokens"],
                (it["end_time"] - ftt) if ftt else -1.0), flush=True)
            if it["token_times"]:
                base = it["token_times"][0]
                gaps = " ".join("%.0f" % ((t - base) * 1000.0) for t in it["token_times"])
                print("[toktime] id=%d 首个token@%.3fs 各token相对首字的毫秒: %s" % (
                    it["request_id"], base - batch_start, gaps), flush=True)
    total_output_tokens = sum(item["output_tokens"] for item in requests)
    ttfts = [
        item["first_token_time"] - item["start_time"]
        for item in requests
        if item["first_token_time"] is not None
    ]
    tpops = [
        (item["end_time"] - item["first_token_time"]) / (item["output_tokens"] - 1)
        for item in requests
        if item["first_token_time"] is not None and item["output_tokens"] > 1
    ]
    request_speeds = [
        item["output_tokens"] / max(item["end_time"] - item["start_time"], 1e-9)
        for item in requests
    ]
    decode_tokens = sum(max(item["output_tokens"] - 1, 0) for item in requests)
    first_token_times = [
        item["first_token_time"] for item in requests
        if item["first_token_time"] is not None
    ]
    batch_decode_span = (
        max(item["end_time"] for item in requests) - min(first_token_times)
        if first_token_times else 0.0
    )
    common_start, common_span, common_tokens = _common_decode_window(
        requests, batch_end)
    last_ttft = max(first_token_times) if first_token_times else None
    decode_before_last = _decode_tokens_before_last_ttft(requests, last_ttft)
    if os.environ.get("FASTLLM_BENCH_TIME_TRACE", "0") not in ("", "0"):
        print("[batchtime] batch_end=%.3fs 最早首字@%.3fs 最晚首字@%.3fs 首字散布=%.3fs" % (
            batch_end - batch_start,
            (min(first_token_times) - batch_start) if first_token_times else -1.0,
            (max(first_token_times) - batch_start) if first_token_times else -1.0,
            (max(first_token_times) - min(first_token_times)) if first_token_times else -1.0),
            flush=True)
    for item, before in zip(requests, decode_before_last):
        item["common_decode_tokens_per_second"] = _per_request_common_rate(
            item, common_start)
        item["decode_tokens_before_last_ttft"] = before
    return {
        "label": label,
        "input_tokens": len(input_tokens),
        "target_output_tokens": output_tokens,
        "batch": batch,
        "requests": requests,
        "total_output_tokens": total_output_tokens,
        "total_time": batch_end - batch_start,
        "ttft_avg": statistics.mean(ttfts) if ttfts else None,
        "ttft_min": min(ttfts) if ttfts else None,
        "ttft_max": max(ttfts) if ttfts else None,
        "tpop_avg": statistics.mean(tpops) if tpops else None,
        "tpop_min": min(tpops) if tpops else None,
        "tpop_max": max(tpops) if tpops else None,
        "per_request_tokens_per_second_avg": (
            statistics.mean(request_speeds) if request_speeds else 0.0
        ),
        "prefill_tokens_per_second": (
            len(input_tokens) / max(ttfts[0], 1e-9) if batch == 1 and ttfts else None
        ),
        "batch_tokens_per_second": total_output_tokens / max(batch_end - batch_start, 1e-9),
        "batch_decode_tokens_per_second": (
            decode_tokens / max(batch_decode_span, 1e-9) if batch_decode_span > 0 else 0.0
        ),
        "common_decode_start": common_start,
        "common_decode_span": common_span,
        "common_decode_tokens": common_tokens,
        "common_decode_tokens_per_second": (
            common_tokens / common_span if common_span > 0 else 0.0
        ),
        "decode_tokens_before_last_ttft": decode_before_last,
        "token_hash": _token_stream_hash(requests),
    }


def _format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.2f} ms"


def _format_ms_per_token(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.2f} ms/token"


def _format_tokens_per_second(value: float) -> str:
    return f"{value:.2f} tokens/s"


def _format_generation_args(generation_args: Dict[str, object]) -> str:
    return (
        f"sample={str(generation_args['do_sample']).lower()}, "
        f"top_p={generation_args['top_p']}, "
        f"top_k={generation_args['top_k']}, "
        f"temperature={generation_args['temperature']}, "
        f"repeat_penalty={generation_args['repeat_penalty']}"
    )


def _print_kv(label: str, value):
    print(f"  {label:<30} {value}")


def _finish_code_message(finish_code: int) -> str:
    if finish_code == -2:
        return "prompt too long"
    return f"generation failed with finish code {finish_code}"


def _print_header(title: str):
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_start(args, generation_args: Dict[str, object], input_tokens: List[int]):
    _print_header("FastLLM Benchmark")
    print("Config")
    _print_kv("Model", args.path or args.model)
    _print_kv("Input tokens", len(input_tokens))
    _print_kv("Output tokens", args.output_tokens)
    _print_kv("Batch", args.batch)
    _print_kv("Max batch", args.max_batch)
    _print_kv("Generation", _format_generation_args(generation_args))
    print()


def _print_result(result: Dict[str, object]):
    _print_header("FastLLM Benchmark Result")
    print("Summary")
    _print_kv("Input tokens", result["input_tokens"])
    _print_kv("Target output tokens", result["target_output_tokens"])
    _print_kv("Batch", result["batch"])
    _print_kv("Actual output tokens", result["total_output_tokens"])
    _print_kv("Total time", f"{result['total_time']:.4f} s")

    print()
    print("Latency")
    _print_kv("TTFT avg", _format_ms(result["ttft_avg"]))
    _print_kv("TTFT min", _format_ms(result["ttft_min"]))
    _print_kv("TTFT max", _format_ms(result["ttft_max"]))
    _print_kv("TPOP avg", _format_ms_per_token(result["tpop_avg"]))
    _print_kv("TPOP min", _format_ms_per_token(result["tpop_min"]))
    _print_kv("TPOP max", _format_ms_per_token(result["tpop_max"]))

    print()
    print("Throughput")
    if result["batch"] == 1 and result["prefill_tokens_per_second"] is not None:
        _print_kv("Prefill", _format_tokens_per_second(result["prefill_tokens_per_second"]))
    _print_kv("Batch total", _format_tokens_per_second(result["batch_tokens_per_second"]))
    _print_kv("Batch decode after TTFT",
              _format_tokens_per_second(result["batch_decode_tokens_per_second"]))
    _print_kv("Batch decode (common window)",
              _format_tokens_per_second(result["common_decode_tokens_per_second"]))
    if result["batch"] > 1 and result["common_decode_start"] is not None:
        # Make the window auditable: without it the two decode rows look like
        # a contradiction rather than a definition difference.
        _print_kv("  common window", "last TTFT %.2f s + %.2f s, %d tokens" % (
            result["common_decode_start"] - min(
                item["start_time"] for item in result["requests"]),
            result["common_decode_span"],
            result["common_decode_tokens"],
        ))
        for item in result["requests"]:
            _print_kv("  request #%d in window" % item["request_id"],
                      _format_tokens_per_second(
                          item["common_decode_tokens_per_second"]))
        before = result.get("decode_tokens_before_last_ttft") or []
        for item, count in zip(result["requests"], before):
            _print_kv("  request #%d before last TTFT" % item["request_id"],
                      "%d tokens" % count)
    _print_kv("Per request avg",
              _format_tokens_per_second(result["per_request_tokens_per_second_avg"]))

    print()
    print("Reproducibility")
    # The digest covers every generated token id, so two runs only compare equal
    # when their per-request token counts match. Print the count on the same line
    # as the digest: comparing digests across different token counts is the exact
    # mistake that produced a phantom "sha drifted between binaries" report.
    _print_kv("Token stream sha256", "%s (tokens/request: %s)" % (
        result["token_hash"],
        ",".join(str(len(item.get("token_ids", [])))
                 for item in result["requests"])))

    early_finished = [
        item for item in result["requests"]
        if item["output_tokens"] < result["target_output_tokens"]
    ]
    errors = [
        item for item in result["requests"]
        if item["finish_code"] is not None and item["finish_code"] != -1
    ]

    if errors:
        print()
        print("Errors")
        for item in errors[:10]:
            _print_kv(f"Request #{item['request_id']}",
                      _finish_code_message(item["finish_code"]))
        if len(errors) > 10:
            _print_kv("More errors", len(errors) - 10)
    elif early_finished:
        print()
        print("Warnings")
        _print_kv("Early finished requests",
                  f"{len(early_finished)} / {result['batch']}")
    print("=" * 72)


def fastllm_benchmark(args):
    _validate_args(args)
    if getattr(args, "max_batch", -1) <= 0:
        args.max_batch = args.batch

    model = make_normal_llm_model(args)
    try:
        generation_args = _generation_args(model, args)
        input_tokens = _build_input_tokens(model, args.input_tokens, args.prompt_unit)

        _print_start(args, generation_args, input_tokens)

        for warmup_index in range(args.warmup):
            print(f"Warmup {warmup_index + 1}/{args.warmup} ...")
            _run_batch(model, input_tokens, min(args.output_tokens, 8), 1,
                       generation_args, label="warmup")
        if args.warmup > 0:
            print()

        result = _run_batch(model, input_tokens, args.output_tokens, args.batch,
                            generation_args,
                            stagger_s=float(getattr(args, "stagger_s", 0.0) or 0.0))
        _print_result(result)
        return result
    finally:
        model.release_memory()


if __name__ == "__main__":
    fastllm_benchmark(args_parser())
