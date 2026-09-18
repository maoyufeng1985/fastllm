#!/usr/bin/env python3
"""loadtest_server.py -- 打生产服务端的 HTTP 接口。

4 条【不同】prompt（前缀缓存不会互相命中），每条 ~16K token，错开 5 秒发射，
每条最多 256 输出 token。用流式读取，因为只有流式才能测到首字时刻。

语料说明（v2）：早先版本用"同一句话重复 696 遍"的填充文本，模型几百 token 内
就输出结束符，测不到 256。现在改为**从仓库取四组各不相同的真实文本**（文档 /
CUDA 源码 / Python 工具 / 头文件），并在末尾追加一个要求长输出的任务指令，
让模型能真正写满 max_tokens。每条都会打印 finish_reason：'length' 才代表跑满。

用法:
  PYTHONPATH=/home/fastllm/build-sm70-tests/tools python3 tools/loadtest_server.py

产出: /tmp/loadtest_server.json（结构化结果）+ stdout 表格。
"""
import glob
import json
import threading
import time
import urllib.request
from ftllm.llm import tokenizer

API = "http://127.0.0.1:8080/v1/chat/completions"
KEY = "maoyufeng1985"
MODEL = "/home/models/Qwen3.8-27B-QUASAR-NVFP4"
MODEL_NAME = "Qwen3.8-27B"

N_TASKS = 4
PROMPT_TOKENS = 16000
MAX_TOK = 256
STAGGER = 5.0
TIMEOUT = 900

# 四组互不相同的真实语料（相对仓库根目录）
CORPUS = [
    ("文档",   sorted(glob.glob("docs/*.md"))),
    ("CUDA 源码", sorted(glob.glob("src/devices/cuda/**/*.cu", recursive=True))),
    ("Python 工具", sorted(glob.glob("tools/fastllm_pytools/**/*.py", recursive=True))),
    ("C++ 头文件", sorted(glob.glob("include/**/*.h", recursive=True))),
]
INSTRUCTION = (
    "\n\n以上是{kind}。请写一份详细的技术总结：分成 8 个小节，"
    "每节至少 120 字，逐节展开论述，包含具体的技术细节，不要省略、不要提前结束。"
)

tk = tokenizer(MODEL)
T0 = time.perf_counter()
results = {}
lock = threading.Lock()


def build_prompt(i):
    kind, files = CORPUS[i]
    parts, used, ntok = [], 0, 0
    for f in files:
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        t = len(tk.encode(text))
        if ntok + t > PROMPT_TOKENS:
            # 最后一个文件按需截断（按字符比例粗截，再用 tokenizer 校正）
            need = PROMPT_TOKENS - ntok
            cut = max(1, int(len(text) * need / max(t, 1)))
            text = text[:cut]
            t = len(tk.encode(text))
        parts.append("# 文件: %s\n%s" % (f, text))
        ntok += t
        used += 1
        if ntok >= PROMPT_TOKENS:
            break
    doc = "\n".join(parts)
    prompt = doc + INSTRUCTION.format(kind=kind)
    return prompt, len(tk.encode(prompt)), used


def run(i):
    prompt, ntok, nfiles = build_prompt(i)
    time.sleep(STAGGER * i)                      # 错开入场
    launch_abs = time.perf_counter()
    launch = launch_abs - T0
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOK,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + KEY,
    })
    rec = {"id": i, "launch": launch, "prompt_tokens_sent": ntok, "files": nfiles,
           "ttft": None, "first_token_at": None, "end": None,
           "tokens": 0, "token_times": [], "finish_reason": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw in r:
                raw = raw.strip()
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:]
                if payload == b"[DONE]":
                    break
                d = json.loads(payload)
                if d.get("usage"):
                    rec["usage"] = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if rec["ttft"] is None:
                        rec["ttft"] = now - launch_abs
                        rec["first_token_at"] = now - T0
                    rec["tokens"] += 1
                    rec["token_times"].append(now - T0)
    except Exception as e:                        # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    rec["end"] = time.perf_counter() - T0
    with lock:
        results[i] = rec


threads = [threading.Thread(target=run, args=(i,)) for i in range(N_TASKS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("=" * 92)
print("生产服务端压测 v2：%d 条不同真实语料 × ~%d token，错开 %.0fs，max_tokens=%d"
      % (N_TASKS, PROMPT_TOKENS, STAGGER, MAX_TOK))
print("=" * 92)
print("%-4s %-10s %-9s %-10s %-12s %-9s %-9s %-8s %-12s" %
      ("id", "语料", "发射(s)", "首字(s)", "首字延迟(s)", "结束(s)", "输出tok", "解码(s)", "finish"))
for i in range(N_TASKS):
    r = results.get(i) or {}
    if r.get("error"):
        print("%-4d ERROR %s" % (i, r["error"]))
        continue
    dec = (r["end"] - r["first_token_at"]) if r["first_token_at"] else 0.0
    print("%-4d %-10s %-9.3f %-10s %-12s %-9.3f %-9d %-8.3f %-12s" % (
        i, CORPUS[i][0], r["launch"],
        ("%.3f" % r["first_token_at"]) if r["first_token_at"] else "None",
        ("%.3f" % r["ttft"]) if r["ttft"] is not None else "None",
        r["end"], r["tokens"], dec, r.get("finish_reason")))
ok = [r for r in results.values() if not r.get("error")]
if ok:
    ends = [r["end"] for r in ok]
    fts = [r["first_token_at"] for r in ok if r["first_token_at"]]
    print("-" * 92)
    print("总时长(最后一条结束)= %.3f s   首字散布= %.3f s"
          % (max(ends), (max(fts) - min(fts)) if fts else -1))
    # 稳态解码：每条取 token 间隔的后半段中位数，避开启动阶段被挤压的部分
    import statistics
    meds = []
    for r in ok:
        ts = r["token_times"]
        if len(ts) >= 4:
            gaps = [(ts[j + 1] - ts[j]) * 1000 for j in range(len(ts) - 1)]
            half = gaps[len(gaps) // 2:]
            meds.append((r["id"], statistics.median(half)))
    if meds:
        print("稳态解码（每条取后半段中位间隔）：")
        for rid, m in meds:
            print("  id=%d 中位间隔 %.1f ms -> 单条 %.1f tok/s" % (rid, m, 1000 / m))
        mm = statistics.median([m for _, m in meds])
        print("  中位 %.1f ms => 单条 %.1f tok/s，%d 条合计 %.0f tok/s"
              % (mm, 1000 / mm, len(meds), len(meds) * 1000 / mm))
    for r in ok:
        if r.get("usage"):
            print("  id=%d 服务端计账: prompt=%s completion=%s" %
                  (r["id"], r["usage"].get("prompt_tokens"),
                   r["usage"].get("completion_tokens")))
# 结果文件按运行时刻命名：早先固定写 /tmp/loadtest_server.json，三次不同配置的跑
# 互相覆盖，基线数据丢了，只能从 stdout 重新解析。现在带时间戳，不再覆盖。
import datetime
OUT = "/tmp/loadtest_server_%s.json" % datetime.datetime.now().strftime("%m%d_%H%M%S")
with open(OUT, "w") as f:
    json.dump({"results": results}, f, indent=1)
print("结果已写入 %s" % OUT)
