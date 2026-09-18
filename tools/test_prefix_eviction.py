#!/usr/bin/env python3
"""test_prefix_eviction.py -- 前缀缓存被挤出后，续写还命不命中？

要回答的问题：历史缓存（--cache_history）在这台机器上有没有活干？
  历史缓存只在"前缀被挤出分页缓存"时才有用。本机分页池 668288 token，
  单会话 16K，理论要 ~42 个不同会话才能把它挤满。本脚本就压到那个量级，
  然后回头再发一次第一个会话的续写，看 cached_tokens 还在不在。

不需要改任何引擎代码：命中情况由服务端返回的 cached_tokens 直接给出。

三段：
  M0 对照   单会话走一轮 首轮->续写，确认"低压力下确实命中"（证明这把尺子能测出真假）
  M1 施压   K 个互不相同的 16K 会话，并发 8，把池子推过 668288 token
  M2 复测   再发一次会话 0 的续写，看 cached_tokens

判据（跑前定死）：
  C1 三段都跑完，M1 的成功请求数 == K
  C2 M0 必须命中（cached_tokens > 0），否则这把尺子不可信，M2 的结论作废
  C3 M1 推入的总 token 数 > 池子容量 668288（算术自检：真的构成压力）

结论怎么读：
  M2 cached_tokens 仍接近 16000 -> 前缀没被挤出 -> 历史缓存在这台机器上没有活干
  M2 cached_tokens 为 0        -> 前缀被挤出了   -> 历史缓存有活干
"""
import glob
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "maoyufeng1985"
MODEL = "/home/models/Qwen3.8-27B-QUASAR-NVFP4"
MODEL_NAME = "Qwen3.8-27B"

TEXT_TOKENS = 15900          # 加模板 -> prompt 16000（125 页整，页对齐）
MAX_TOK = 8                  # 只要够触发一次记录，不要解码时间
K_SESSIONS = int(os.environ.get("K_SESSIONS", "56"))
CONC = int(os.environ.get("CONC", "8"))
POOL_TOKENS = 668288         # 服务端启动日志里报的分页池容量

if os.environ.get("DRYRUN") == "1":
    print("[DRYRUN] 不碰显卡，只报将要做什么")
    print("  目标      :", API)
    print("  模型      :", MODEL_NAME)
    print("  会话长度  : %d token/条" % (TEXT_TOKENS + 100))
    print("  施压会话数: %d（并发 %d）" % (K_SESSIONS, CONC))
    print("  推入总量  : %d token vs 池子 %d token -> %s"
          % (K_SESSIONS * 16000, POOL_TOKENS,
             "够构成压力" if K_SESSIONS * 16000 > POOL_TOKENS else "不够，需调大 K_SESSIONS"))
    print("  请求总数  : %d（M0 两发 + M1 %d 发 + M2 一发）" % (K_SESSIONS + 3, K_SESSIONS))
    print("  预算时长  : 约 %.1f 分钟（按每波 16K 预填 ~12s 估）"
          % (K_SESSIONS / CONC * 12 / 60 + 0.5))
    sys.exit(0)

from ftllm.llm import tokenizer     # noqa: E402

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


print("  正在拼语料 ...")
parts = []
for pat in ("docs/*.md", "tools/*.py", "src/models/*.cpp"):
    for f in sorted(glob.glob(pat)):
        try:
            parts.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
CORPUS = "\n".join(parts)
print("  语料字符数 =", len(CORPUS))

# 每条会话加不同的头 + 从不同偏移开始，保证 token 序列从第 0 位就分叉，
# 各自占各自的页，不会被分页缓存合并
_prompt_cache = {}
# 先切一个固定大小的窗口再做二分，不要把整个语料丢给 fit_tokens：
# 那样每个会话都要对几 MB 的字符串做一次编码（实测烧了 44s CPU、显卡一次没碰）
WINDOW = 120000


def prompt_for(i):
    if i not in _prompt_cache:
        hdr = "[片段%03d] " % i
        start = (i * 7919) % max(1, len(CORPUS) - WINDOW)
        _prompt_cache[i] = fit_tokens(hdr + CORPUS[start:start + WINDOW], TEXT_TOKENS)
    return _prompt_cache[i]


def chat(messages, timeout=900):
    body = json.dumps({"model": MODEL_NAME, "messages": messages,
                       "max_tokens": MAX_TOK, "temperature": 0}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {}) or {}
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    return dt, (det.get("cached_tokens") or 0), ans


def continuation(i, answer):
    return [{"role": "user", "content": prompt_for(i)},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "继续"}]


print("=" * 78)
print("前缀缓存挤出实验   会话长度 %d token   施压 %d 条   并发 %d"
      % (TEXT_TOKENS + 100, K_SESSIONS, CONC))
print("=" * 78)

print("  正在预造 %d 条会话的 prompt ..." % (K_SESSIONS + 1))
_t0 = time.perf_counter()
for _i in range(0, K_SESSIONS + 1):
    prompt_for(_i)
    if (_i + 1) % 10 == 0:
        print("    已造 %d/%d  (%.1fs)" % (_i + 1, K_SESSIONS + 1, time.perf_counter() - _t0))
print("  造完 %d 条，用时 %.1fs，首条 token 数=%d"
      % (K_SESSIONS + 1, time.perf_counter() - _t0, len(tk.encode(prompt_for(0)))))

# ---- M0 对照 ----
t1, c1, a0 = chat([{"role": "user", "content": prompt_for(0)}])
print("  M0 会话0首轮       %6.3fs  cached=%s" % (t1, c1))
t2, c2, _ = chat(continuation(0, a0))
print("  M0 会话0续写       %6.3fs  cached=%s   <-- 必须 > 0，否则尺子不可信" % (t2, c2))

# ---- M1 施压 ----
print("-" * 78)
print("  M1 开始施压：%d 条互不相同的 16K 会话 ..." % K_SESSIONS)
t_start = time.perf_counter()
ok, fail, pushed = 0, 0, 0
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = {ex.submit(chat, [{"role": "user", "content": prompt_for(i)}]): i
            for i in range(1, K_SESSIONS + 1)}
    for n, fut in enumerate(futs, 1):
        i = futs[fut]
        try:
            dt, cv, _ = fut.result()
            ok += 1
            pushed += 16000
            if n % 8 == 0 or n == len(futs):
                print("    进度 %3d/%d  最近一条 %6.2fs cached=%s  已推入 %d token"
                      % (n, len(futs), dt, cv, pushed))
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print("    !! 会话 %d 失败: %s" % (i, e))
print("  M1 结束：成功 %d 失败 %d 用时 %.1fs 推入 %d token"
      % (ok, fail, time.perf_counter() - t_start, pushed))

# ---- M2 复测 ----
print("-" * 78)
t3, c3, _ = chat(continuation(0, a0))
print("  M2 会话0续写(复测) %6.3fs  cached=%s" % (t3, c3))

print("=" * 78)
c1_ok = c2 > 0
c3_ok = ok == K_SESSIONS
c4_ok = pushed > POOL_TOKENS
print("判据 C1（三段跑完，施压成功 %d/%d）: %s" % (ok, K_SESSIONS, "通过" if c3_ok else "未通过"))
print("判据 C2（M0 对照命中，尺子可信）  : %s (cached=%s)" % ("通过" if c1_ok else "未通过", c2))
print("判据 C3（推入量真的构成压力）     : %s (%d vs %d)"
      % ("通过" if c4_ok else "未通过", pushed, POOL_TOKENS))
print("-" * 78)
if not (c1_ok and c4_ok):
    print("结论：**本轮作废**（判据未过，不能解读 M2）")
elif c3 > 0:
    print("结论：前缀没被挤出（M2 cached=%d）-> 历史缓存在这台机器上没有活干" % c3)
else:
    print("结论：前缀已被挤出（M2 cached=0）-> 历史缓存有活干")

out = "/tmp/prefix_eviction_%s.json" % time.strftime("%m%d_%H%M%S")
json.dump({"m0": {"first": t1, "cont": t2, "cached": c2},
           "m1": {"sessions": K_SESSIONS, "ok": ok, "fail": fail, "pushed": pushed,
                  "seconds": time.perf_counter() - t_start},
           "m2": {"cont": t3, "cached": c3},
           "pool_tokens": POOL_TOKENS,
           "verdict": {"c2_control_ok": c1_ok, "c3_pressure_ok": c4_ok,
                       "prefix_survived": c3 > 0}},
          open(out, "w"), ensure_ascii=False, indent=1)
print("结果已写入", out)
