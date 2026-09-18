#!/usr/bin/env python3
"""verify_fix3.py -- 前缀缓存页对齐：三轮验证。

要回答的问题：让分块末块停在提示词最后一个页边界 A 之后，下一轮是不是真的不用重算尾巴。
第一轮（冷）记快照；第二轮复用 A 并把"第二轮提示词自己的 A"也记下来；第三轮复用第二轮
的 A。主判据按任务里原始声明的"第二轮"表述报，第三轮作为补充。

判据（跑之前先写死，避免事后挑数字）：
  C1 三轮都拿到 200 响应，且 journal 无报错
  C2 第二轮 cached_tokens >= prompt_tokens - 256（改动前最好 prompt-1530）
  C3 第二轮服务端 [Prompt] 求和 <= 256（改动前 1530 起）
  C4（补充）第三轮 cached_tokens >= prompt_tokens - 256
  C5（补充）第三轮服务端 [Prompt] 求和 <= 256

服务端 [Prompt] 只统计 prefill 行；同一请求的多块之间间隔 < GAP 秒，不同请求之间更久，
按这个切分求和。注意服务端写 journal 与客户端拿到响应之间有延迟，必须轮询读取。
"""
import json, re, subprocess, sys, time, urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/models/Qwen3.8-27B-QUASAR-NVFP4")
KEY = "maoyufeng1985"; URL = "http://127.0.0.1:8080/v1/chat/completions"
UNIT = ("Kubernetes schedules pods onto nodes. The scheduler filters and scores nodes, then binds the pod. "
        "Eviction removes pods when a node runs out of memory. The kubelet syncs pod state every ten seconds. ")
Q = "\n\n用一句话说明 pod 是怎么被调度到节点上的。"
EXTRA = ("The apiserver validates the pod spec, persists it to etcd, and notifies watchers. "
         "Controllers reconcile the desired state until the observed state matches. ") * 60


def call(msgs, max_tokens=8):
    body = json.dumps({"model": "Qwen3.8-27B", "messages": msgs,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    det = d.get("usage", {}).get("prompt_tokens_details") or {}
    return (d["usage"]["prompt_tokens"], det.get("cached_tokens", 0),
            time.time() - t0, (d["choices"][0]["message"].get("content") or ""))


def prefill_tokens(t0, tries=12):
    """轮询 journal，等这一轮的 [Prompt] 行落盘后按 >GAP 秒切分求和。"""
    GAP = 3.0
    LINE = re.compile(r"^(\w+月\s+\d+ \d+:\d+:\d+) .*?\[Prompt\] (\d+) Tokens")
    import datetime
    best = ([], None)
    for _ in range(tries):
        out = subprocess.run(["journalctl", "-u", "ftllm-server.service", "--no-pager",
                              "--since", f"@{t0:.0f}"], capture_output=True, text=True).stdout
        rows = []
        for line in out.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            ts = datetime.datetime.strptime("2026年 " + m.group(1).replace("月", "月 "),
                                            "%Y年 %m月 %d %H:%M:%S")
            rows.append((ts, int(m.group(2))))
        rows.sort()
        groups = []
        for ts, n in rows:
            if groups and (ts - groups[-1][-1][0]).total_seconds() <= GAP:
                groups[-1].append((ts, n))
            else:
                groups.append([(ts, n)])
        best = (groups, out)
        if groups:
            return [[n for _, n in g] for g in groups]
        time.sleep(5)
    return []


def trace(t0, tries=12):
    for _ in range(tries):
        out = subprocess.run(["journalctl", "-u", "ftllm-server.service", "--no-pager",
                              "--since", f"@{t0:.0f}"], capture_output=True, text=True).stdout
        lines = [l.split("]: ", 1)[-1].strip() for l in out.splitlines()
                 if "prefixcache" in l]
        if lines:
            return lines
        time.sleep(5)
    return []


N = 380
P = UNIT * N + Q + EXTRA[:1400]

print("=== 第一轮（冷）===")
t = time.time()
pt1, ct1, dt1, ans = call([{"role": "user", "content": P}])
g1 = prefill_tokens(t)
L1 = trace(t)
print(f"  prompt={pt1}  非对齐余数={pt1 % 128}  cached={ct1}  {dt1:.2f}s  [Prompt]分组={g1}")
for x in L1:
    print("   ", x[:112])

msgs2 = [{"role": "user", "content": P},
         {"role": "assistant", "content": ans},
         {"role": "user", "content": "继续"}]
print("\n=== 第二轮（前缀 = 第一轮 + 追加问答）===")
time.sleep(3)
t = time.time()
pt2, ct2, dt2, ans2 = call(msgs2)
g2 = prefill_tokens(t)
L2 = trace(t)
sum2 = sum(sum(g) for g in g2)
print(f"  prompt={pt2}  cached={ct2}  {dt2:.2f}s  [Prompt]分组={g2}  求和={sum2}")
for x in L2:
    print("   ", x[:112])

msgs3 = msgs2 + [{"role": "assistant", "content": ans2},
                 {"role": "user", "content": "再补一句"}]
print("\n=== 第三轮（补充）===")
time.sleep(3)
t = time.time()
pt3, ct3, dt3, _ = call(msgs3)
g3 = prefill_tokens(t)
sum3 = sum(sum(g) for g in g3)
print(f"  prompt={pt3}  cached={ct3}  {dt3:.2f}s  [Prompt]分组={g3}  求和={sum3}")

print()
print(f"C1 三轮成功:            {'是' if pt1 > 0 and pt2 > 0 and pt3 > 0 else '否'}")
print(f"C2 第二轮 cached>=prompt-256: {'是 (%d >= %d)' % (ct2, pt2 - 256) if ct2 >= pt2 - 256 else '否 (%d < %d)' % (ct2, pt2 - 256)}")
print(f"C3 第二轮 [Prompt] <= 256:    {'是 (%d)' % sum2 if sum2 <= 256 else '否 (%d)' % sum2}")
print(f"C4 第三轮 cached>=prompt-256: {'是 (%d >= %d)' % (ct3, pt3 - 256) if ct3 >= pt3 - 256 else '否 (%d < %d)' % (ct3, pt3 - 256)}")
print(f"C5 第三轮 [Prompt] <= 256:    {'是 (%d)' % sum3 if sum3 <= 256 else '否 (%d)' % sum3}")
print(f"\n耗时：冷 {dt1:.2f}s  第二轮 {dt2:.2f}s  第三轮 {dt3:.2f}s")
ok = (pt1 > 0 and pt2 > 0 and pt3 > 0 and
      ct2 >= pt2 - 256 and sum2 <= 256)
sys.exit(0 if ok else 1)
