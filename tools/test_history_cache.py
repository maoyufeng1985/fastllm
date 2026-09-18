#!/usr/bin/env python3
"""test_history_cache.py -- 测 history cache（--cache_history true，走 CPU）是否生效。

设计：同一段 ~16K prompt 连发三次，看服务端计账的 cached_tokens 与首字延迟：
  第 1 次 冷启动   -> cached_tokens 应≈0，首字最慢
  第 2 次 完全相同 -> cached_tokens 应≈全量，首字最快
  第 3 次 加长续写 -> 部分命中（cached 介于两者之间）

同时在每轮前后读服务进程的 RSS：历史 KV 若真的落在 CPU，主机侧内存应增长。
判据（跑之前定死）：
  H1 cached_tokens 单调：冷 < 第3次 ≤ 第2次
  H2 进程 RSS 在首轮之后增长（CPU 侧真的存了东西）
  H3 三次结果都正常返回（不因缓存出错）
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
PROMPT_TOKENS = 16000
TARGET_BASE_TOK = 16028   # prompt 总数 16128 = 126*128（页对齐）
MAX_TOK = 1            # 极短生成：让 prefill 结束那次 record 的 currentLen == prompt 本身

tk = tokenizer(MODEL)


def build_base():
    files = sorted(glob.glob("docs/*.md"))
    parts, ntok = [], 0
    for f in files:
        t = open(f, encoding="utf-8", errors="ignore").read()
        n = len(tk.encode(t))
        if ntok + n > PROMPT_TOKENS:
            need = PROMPT_TOKENS - ntok
            t = t[:max(1, int(len(t) * need / max(n, 1)))]
            n = len(tk.encode(t))
        parts.append(t)
        ntok += n
        if ntok >= PROMPT_TOKENS:
            break
    return "\n".join(parts)



def fit_tokens(text, target):
    """把 text 截/补到恰好 target 个 token（tokenizer 无 decode，二分字符长度逼近）。"""
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tk.encode(text[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    out = text[:lo]
    n = len(tk.encode(out))
    while n < target:                       # 差几个就补几个单 token 字符
        out += "。"
        n = len(tk.encode(out))
        if len(out) > len(text) + 64:
            break
    while n > target and lo > 1:            # 多了就再退
        lo -= 1
        out = text[:lo]
        n = len(tk.encode(out))
    return out

BASE = fit_tokens(build_base(), TARGET_BASE_TOK)
BASE_TOK = len(tk.encode(BASE))
TAIL = "\n\n请用一句话总结上面内容。"          # 第 3 次用的加长续写


def ask(prompt, tag):
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {}) or {}
    print("  %-14s 首字+生成 %.3fs  prompt=%s cached=%s missed=%s completion=%s"
          % (tag, dt, u.get("prompt_tokens"), det.get("cached_tokens"),
             det.get("missed_tokens"), u.get("completion_tokens")))
    return {"tag": tag, "wall": dt, "usage": u, "cached": det.get("cached_tokens") or 0}


def rss_mb():
    pid = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value",
                          "ftllm-server.service"], capture_output=True, text=True).stdout.strip()
    try:
        with open("/proc/%s/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


print("=" * 78)
print("history cache 测试：同一段 %d token prompt 发三次（冷 / 重复 / 加长）" % BASE_TOK)
print("=" * 78)
print("  起始 RSS = %.0f MB" % rss_mb())
r1 = ask(BASE, "1 冷启动")
print("  RSS = %.0f MB" % rss_mb())
r2 = ask(BASE, "2 完全相同")                    # 这次应当命中第 1 次记下的快照
print("  RSS = %.0f MB" % rss_mb())
r3 = ask(BASE + TAIL, "3 加长续写")
r4 = ask(BASE, "4 再来一次")
print("  结束 RSS = %.0f MB" % rss_mb())
print("-" * 78)
print("cached_tokens: 冷=%s 重复=%s 加长=%s 再来=%s"
      % (r1["cached"], r2["cached"], r3["cached"], r4["cached"]))
