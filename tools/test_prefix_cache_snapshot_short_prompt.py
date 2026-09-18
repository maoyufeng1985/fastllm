#!/usr/bin/env python3
"""test_prefix_cache_snapshot_short_prompt.py -- 短提示词的 token 有没有被吃掉？

要回答的问题：改动"让分块末块停在提示词最后一个页边界"时，未分块的一次性 prefill
会不会也被截短。会的话，模型拿到的是被静默截断的提示词，不报任何错。

背景（实测过的真实回归）：那个改动一开始只判 `curLen == remaining`，而未分块的请求里
remaining 就等于 currentTokens.size()、prefillRemaining 仍是 0，根本没有"下一轮"去
吃掉 A 之后的尾巴。于是 266 token 的提示词只 prefill 了 128，176 token 的一条
服务端一个 [Prompt] 行都没有。修法是加 `ctx->prefillRemaining > 0` 的限定。

判据（跑之前先写死，避免事后挑数字）：
  C1 每条提示词都返回 200
  C2 窗口里存在一个（按 >GAP 秒切分的）[Prompt] 分组，其求和 == prompt_tokens - cached_tokens
     —— 即"没有命中缓存的那部分 token"全都真的进了 prefill。被截短时不会存在这样的分组。
  C3 全冷（cached_tokens == 0）时，[Prompt] 求和 == prompt_tokens

为什么不能用 usage 里的 missed_tokens 当判据（这是我第一版犯的错）：它由服务端算成
`inputTokens - cacheLen`（src/models/basellm.cpp:3566），是个恒等式，同一次请求里
只要 cacheLen 不变就恒等于它自己，截不截短都一样，验不出问题。
真正有信息量的对照是"引擎自己打印的 prefill token 数"对"API 报的 prompt_tokens 减去命中数"。

为什么用"存在某个分组相等"而不是"窗口求和相等"：这台机器上还有别人在打真实流量，
窗口里会混进无关的 [Prompt] 行。干扰只会让分组变多/变大，不会变小，所以
"存在一个分组等于期望值"对"有没有被截短"是可靠的判据。
注意服务端写 journal 与客户端拿到响应之间有延迟，只读一次会漏（实测漏过），必须轮询。

用法：python3 tools/test_prefix_cache_snapshot_short_prompt.py
前提：服务端跑在 127.0.0.1:8080，KEY 见下；提示词故意取 <= chunked_prefill_size(2048)
      且不是 128 整数倍的长度。每条用随机唯一前缀，避免互相命中缓存。
"""
import datetime
import json
import random
import re
import string
import subprocess
import sys
import time
import urllib.request

KEY = "maoyufeng1985"
URL = "http://127.0.0.1:8080/v1/chat/completions"
GAP = 3.0  # 同一请求的分块间隔 <GAP 秒，不同请求之间更久，按这个切分求和
LINE = re.compile(r"^(\w+月\s+\d+ \d+:\d+:\d+) .*?\[Prompt\] (\d+) Tokens")


def call(text, max_tokens=8):
    """返回 (HTTP 码, prompt_tokens, cached_tokens)。"""
    body = json.dumps({"model": "Qwen3.8-27B",
                       "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
        code = r.status
    usage = d["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return code, usage["prompt_tokens"], details.get("cached_tokens", 0)


def prefill_groups(t_from, want, tries=12):
    """把 t_from 之后的 [Prompt] 行按 >GAP 秒切成"每次请求一组"，返回各组求和列表。

    服务端写 journal 与客户端拿到响应之间有延迟，只读一次会漏（实测漏过），所以轮询。
    """
    groups = []
    for _ in range(tries):
        out = subprocess.run(
            ["journalctl", "-u", "ftllm-server.service", "--no-pager",
             "--since", f"@{int(t_from)}"],
            capture_output=True, text=True).stdout
        rows = []
        for line in out.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            ts = datetime.datetime.strptime(
                "2026年 " + m.group(1).replace("月", "月 "),
                "%Y年 %m月 %d %H:%M:%S")
            rows.append((ts, int(m.group(2))))
        rows.sort()
        groups = []
        for ts, n in rows:
            if groups and (ts - groups[-1][-1][0]).total_seconds() <= GAP:
                groups[-1].append((ts, n))
            else:
                groups.append([(ts, n)])
        sums = [sum(n for _, n in g) for g in groups]
        # 期望的那一组已经落盘就提前收工（新请求通常只有一块）
        if want in sums:
            return sums
        time.sleep(5)
    return [sum(n for _, n in g) for g in groups]


def main():
    # 每轮换种子：不然同一条提示词在下一轮会命中上一轮的快照，
    # 那条就不再是"全冷"，测不到未分块 prefill 这条路径。
    random.seed()
    print("短提示词（唯一前缀，各条互不重叠）:")
    bad = 0
    for target in (116, 176, 266, 567, 1766):
        uniq = "".join(random.choice(string.ascii_lowercase) for _ in range(target * 3))
        text = uniq + "\n\n用一句话说明 pod 是怎么被调度到节点上的。"
        time.sleep(2)
        t0 = time.time()
        code, prompt_tokens, cached = call(text)
        expected = prompt_tokens - cached
        sums = prefill_groups(t0, expected)
        ok = (code == 200 and expected in sums)
        bad += 0 if ok else 1
        print(f"  HTTP={code} prompt_tokens={prompt_tokens:5d} cached={cached:5d} "
              f"应 prefill={expected:5d}  观测到的 [Prompt] 分组={sums}"
              f"  {'一致' if ok else '*** 不一致（尾巴被吃掉?）'}")

    print()
    print(f"C1 全部返回 200:                {'是' if bad == 0 else '否，%d 条异常' % bad}")
    print(f"C2 存在分组 == prompt-cached:   {'是' if bad == 0 else '否'}")
    print("   （C3 全冷时的相等已包含在 C2 里：那几条 cached=0，expected==prompt_tokens）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
