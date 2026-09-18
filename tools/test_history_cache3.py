#!/usr/bin/env python3
"""test_history_cache3.py -- 用真正的多轮对话形态验证前缀/历史缓存命中。

前几轮的教训（都已在追踪里看到证据）：
  - 快照只在 currentLen 是页大小(128)整数倍时被记录；prompt=16000(125页) 可以
  - 匹配条件是 qwen3_5.cpp:5220 的 std::equal：快照的 token 序列必须与新请求
    **从头逐位相同**
  - 我先前用"同一段文本再发一次、只是更长"去测，那是**另一条 prompt**，
    两者在结尾的生成标记处就分叉了，所以永远配不上

正确形态（多轮对话）：
  turn1: messages=[user: TEXT]
  turn2: messages=[user: TEXT, assistant: ANSWER, user: NEWQ]
  turn2 的 prompt token 序列**恰好以 turn1 的 prompt 为前缀** -> 快照可命中

判据（跑前定死）：
  C1 turn1 之后追踪出现 record OK
  C2 turn2 的 query 不是"无快照"（cached_tokens > 0）
  C3 turn2 的首字明显低于 turn1（缓存带来的实际收益）
"""
import glob
import json
import time
import urllib.request
from ftllm.llm import tokenizer

API = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "maoyufeng1985"
MODEL = "/home/models/Qwen3.8-27B-QUASAR-NVFP4"
MODEL_NAME = "Qwen3.8-27B"
TEXT_TOKENS = 15900          # 加模板 -> prompt 16000（125 页整，页对齐）
MAX_TOK = 32

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
    return out


parts, n = [], 0
for f in sorted(glob.glob("docs/*.md")):
    t = open(f, encoding="utf-8", errors="ignore").read()
    parts.append(t); n += len(tk.encode(t))
    if n > 20000:
        break
TEXT = fit_tokens("\n".join(parts), TEXT_TOKENS)
print("  正文 token 数 =", len(tk.encode(TEXT)))


def chat(messages, tag):
    body = json.dumps({"model": MODEL_NAME, "messages": messages,
                       "max_tokens": MAX_TOK, "temperature": 0}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {}) or {}
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    print("  %-18s 耗时 %.3fs prompt=%s cached=%s missed=%s 回答=%r"
          % (tag, dt, u.get("prompt_tokens"), det.get("cached_tokens"),
             det.get("missed_tokens"), ans[:28]))
    return dt, det.get("cached_tokens") or 0, ans


print("=" * 78)
print("多轮对话形态的前缀缓存验证")
print("=" * 78)
t1, c1, a1 = chat([{"role": "user", "content": TEXT}], "1 首轮(记快照)")
t2, c2, a2 = chat([{"role": "user", "content": TEXT},
                   {"role": "assistant", "content": a1},
                   {"role": "user", "content": "继续"}], "2 续写(应命中)")
t3, c3, a3 = chat([{"role": "user", "content": TEXT},
                   {"role": "assistant", "content": a1},
                   {"role": "user", "content": "继续"}], "3 再续一次")
print("-" * 78)
print("cached_tokens: 首轮=%s 续写=%s 再续=%s" % (c1, c2, c3))
print("首字: %.3f / %.3f / %.3f s" % (t1, t2, t3))
print("判据 C2（续写应命中缓存）: %s" % ("通过" if c2 > 0 else "未通过"))
print("判据 C3（续写应更快）: %s"
      % ("通过（%.1f%%）" % ((t2 / t1 - 1) * 100) if t2 < t1 * 0.8
         else "未通过（%.1f%%）" % ((t2 / t1 - 1) * 100)))
