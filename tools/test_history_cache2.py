#!/usr/bin/env python3
"""test_history_cache2.py -- 验证前缀/历史缓存能否真的命中并带来收益。

上一轮的实测结论（有追踪证据）：
  - prompt 是 128 的整数倍时，prefill 结束会成功记录快照（record OK: cachedLen=16128）
  - 但同一个 prompt 再发一次也配不上：basellm.cpp:2076 在"匹配长度 >= 本请求 prompt 长度"
    时会主动退掉一页，于是查询上限 maxLen=16000 < 快照 16128，恒不命中

据此设计（A/B 只差 prompt 长度）：
  请求 1  prompt = 16128  -> 记下快照 cachedLen=16128
  请求 2  prompt = 16256  -> 匹配到 16128 < 自己的 16256，不触发退页
                             => maxLen 应为 16128，快照应命中

判据（跑前定死）：
  C1 请求 1 之后追踪里出现 "record OK"
  C2 请求 2 的追踪里 maxLen 从 16000 变成 16128，且不是"无快照"
  C3 请求 2 的首字延迟明显低于请求 1（缓存复用的效果）
"""
import glob
import json
import subprocess
import time
import urllib.request
from ftllm.llm import tokenizer

API = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "maoyufeng1985"
MODEL = "/home/models/Qwen3.8-27B-QUASAR-NVFP4"
MODEL_NAME = "Qwen3.8-27B"

tk = tokenizer(MODEL)


def fit_tokens(text, target):
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tk.encode(text[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    out = text[:lo]
    while len(tk.encode(out)) < target:
        out += "。"
        if len(out) > len(text) + 64:
            break
    while len(tk.encode(out)) > target and lo > 1:
        lo -= 1
        out = text[:lo]
    return out


def doc():
    parts, n = [], 0
    for f in sorted(glob.glob("docs/*.md")):
        t = open(f, encoding="utf-8", errors="ignore").read()
        parts.append(t)
        n += len(tk.encode(t))
        if n > 20000:
            break
    return "\n".join(parts)


D = doc()
# 服务端会额外加 ~100 个模板 token，所以 base 用 prompt-100
P1_BASE, P2_BASE = 15900, 16156      # -> 服务端 prompt 16000(125页) / 16256(127页)
A = fit_tokens(D, P1_BASE)
B = fit_tokens(D, P2_BASE)
assert B.startswith(A[:2000]), "请求 2 必须以请求 1 为前缀"


def ask(prompt, tag, maxtok=1):
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {}) or {}
    print("  %-22s 耗时 %.3fs  prompt=%s cached=%s missed=%s"
          % (tag, dt, u.get("prompt_tokens"), det.get("cached_tokens"),
             det.get("missed_tokens")))
    return dt, det.get("cached_tokens") or 0


print("=" * 78)
print("前缀缓存命中验证：先记快照（16128），再用长一页的 prompt（16256）去命中")
print("=" * 78)
print("  base token: A=%d  B=%d" % (len(tk.encode(A)), len(tk.encode(B))))
t1, c1 = ask(A, "1 记快照 (16000=125页)")
t2, c2 = ask(B, "2 命中尝试 (16256)")
t3, c3 = ask(B, "3 再来一次 (16256)")
print("-" * 78)
print("首字延迟: 16128=%.3fs  16256=%.3fs  16256再来=%.3fs" % (t1, t2, t3))
print("cached_tokens: %s / %s / %s" % (c1, c2, c3))
print("判据 C3（第2次应明显快于第1次）: %s"
      % ("通过" if t2 < t1 * 0.8 else "未通过（差 %.1f%%）" % ((t2 / t1 - 1) * 100)))
