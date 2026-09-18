#!/usr/bin/env python3
"""生产服务的解码测速探针：对真实 HTTP 服务发一条流式请求，自己计时。
不改引擎、不用日志里的瞬时值——首字时刻和末字时刻都由本脚本记。
用同一段提示词、贪婪解码，所以落地开关前后两次的 token 流哈希必须一致。
用法: prod_decode_probe.py <重复段数> <最多输出 token 数>
"""
import hashlib
import json
import sys
import time
import urllib.request

PARA = (
    "在并行推理系统里，张量并行会把同一层的权重切到多张卡上，"
    "每层算完都要把各卡的部分和加起来，这个动作就是全归约。"
    "全归约的次数随层数走，跟上下文长短无关，所以上下文越长它占的比例越小。"
)
QUESTION = (
    "\n请写一篇不少于八百字的说明文，分二十条编号逐一展开，"
    "每条都要有具体数字和例子，不要提前结束。"
)


def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    prompt = PARA * repeats + QUESTION

    body = json.dumps({
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer maoyufeng1985",
            "Content-Type": "application/json",
        },
    )

    t0 = time.perf_counter()
    first = None
    last = None
    chunks = 0
    pieces = []
    usage = None
    finish = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
                    pieces.append(text)

    t1 = time.perf_counter()
    content = "".join(pieces)
    digest = hashlib.sha256(content.encode()).hexdigest()[:8]
    if first is None or last is None or chunks < 2:
        print("NO_OUTPUT chunks=%d finish=%s" % (chunks, finish))
        return 1
    decode_s = last - first
    print(
        "prompt_chars=%d ttft_ms=%.1f chunks=%d decode_ms=%.1f "
        "decode_tok_s=%.2f total_ms=%.1f finish=%s content_sha=%s completion=%s"
        % (
            len(prompt), (first - t0) * 1000, chunks, decode_s * 1000,
            (chunks - 1) / decode_s if decode_s > 0 else 0.0,
            (t1 - t0) * 1000, finish, digest,
            (usage or {}).get("completion_tokens"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
