#!/usr/bin/env python3
"""prefill_live.py -- 打生产服务测一次 prefill，不停服务。

为什么这样测：ftllm-server.service 正在被用户使用，不能停。
它本身就是最好的测量对象（同一份二进制、同一套参数、正在服务）。
做法与 tools/loadtest_server.py 一致：流式 POST，量首字时刻；
并从 journald 抓这一窗口内的 `[Prompt] N Tokens` / `[Decode] ... Speed:` 行。
同时按 0.5s 采样 nvidia-smi 的 sm% 与 mem%（显存控制器占用率）。

用法:
  python3 tools/prefill_live.py --input_tokens 16384 --out /tmp/pcab/live_16k
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

API = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "maoyufeng1985"
MODEL_NAME = "Qwen3.8-27B"
MODEL_PATH = "/home/models/Qwen3.8-27B-QUASAR-NVFP4"


def build_prompt(target_tokens, corpus_index):
    """用仓库里的真实文本拼一个 ~target_tokens 的提示词。

    刻意换语料：前缀缓存命中会让真实 prefill 变短，测不到冷启动的 prefill。
    """
    sys.path.insert(0, "/home/fastllm/build-sm70-tests/tools")
    from ftllm.llm import tokenizer
    import glob

    corpus = [
        sorted(glob.glob("/home/fastllm/docs/*.md")),
        sorted(glob.glob("/home/fastllm/src/devices/cuda/**/*.cu", recursive=True)),
        sorted(glob.glob("/home/fastllm/include/**/*.h", recursive=True)),
        sorted(glob.glob("/home/fastllm/tools/fastllm_pytools/**/*.py", recursive=True)),
    ][corpus_index % 4]

    tk = tokenizer(MODEL_PATH)
    parts, ntok = [], 0
    for f in corpus:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        t = len(tk.encode(text))
        if ntok + t > target_tokens:
            need = target_tokens - ntok
            cut = max(1, int(len(text) * need / max(t, 1)))
            text = text[:cut]
            t = len(tk.encode(text))
        parts.append("# 文件: %s\n%s" % (f, text))
        ntok += t
        if ntok >= target_tokens:
            break
    return "\n".join(parts), len(tk.encode("\n".join(parts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tokens", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--corpus", type=int, default=0)
    ap.add_argument("--sample", type=float, default=0.5)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    prompt, ntok = build_prompt(args.input_tokens, args.corpus)
    print("LIVE_PROMPT chars=%d approx_tokens=%d corpus=%d"
          % (len(prompt), ntok, args.corpus), flush=True)

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                ts = time.perf_counter()
                for line in out.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 3:
                        samples.append((ts, int(p[0]), int(p[1]), int(p[2])))
            except Exception:
                pass
            stop.wait(args.sample)

    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + KEY,
    })

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.perf_counter()
    ttft = None
    ntokens = 0
    finish = None
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                raw = raw.strip()
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:]
                if payload == b"[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if (ch.get("delta") or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntokens += 1
    except Exception as e:  # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, e)
    wall = time.perf_counter() - t0
    stop.set()
    t.join(timeout=3)

    print("LIVE_RESULT wall=%.4fs ttft=%s out_tokens=%d finish=%s err=%s"
          % (wall, ("%.4f" % ttft) if ttft else "None", ntokens, finish, err),
          flush=True)
    if usage:
        print("LIVE_USAGE %s" % json.dumps(usage), flush=True)

    by_gpu = {}
    for _, idx, sm, mem in samples:
        by_gpu.setdefault(idx, []).append((sm, mem))
    for idx in sorted(by_gpu):
        vals = by_gpu[idx]
        sms = [v[0] for v in vals]
        mems = [v[1] for v in vals]
        print("LIVE_SAMPLE gpu=%d n=%d sm_avg=%.1f sm_max=%d mem_avg=%.1f mem_max=%d"
              % (idx, len(vals), sum(sms) / len(sms), max(sms),
                 sum(mems) / len(mems), max(mems)), flush=True)

    with open(args.out + ".json", "w") as fh:
        json.dump({"approx_tokens": ntok, "ttft": ttft, "wall": wall,
                   "out_tokens": ntokens, "finish": finish, "err": err,
                   "usage": usage,
                   "samples": samples}, fh)


if __name__ == "__main__":
    main()
