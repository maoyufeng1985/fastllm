#!/usr/bin/env python3
"""test_prefix_cache_align.py -- 非页对齐长度的提示词，第二轮能不能复用前缀？

要回答的问题：Qwen3.5 的跨请求前缀缓存，在提示词总长不是 128 的整数倍时生不生效。
快照只能落在页边界上（线性注意力状态是递推量，记不到中间位置），所以记录时机
必须落在某个页对齐的分块边界上，而不是提示词结尾。

判据（跑前定死）：
  C1 两轮都返回 200
  C2 第一轮诊断出现 record OK，cachedLen 是 128 的整数倍且小于提示词总长
  C3 第二轮 cached_tokens > 0，且诊断出现「命中」

需要服务端带 FASTLLM_PREFIX_CACHE_TRACE=1 运行才能看到诊断行。
"""
判据（跑前定死）：
  C1 两轮都拿到 200 响应
  C2 第一轮诊断里出现 record OK，且 cachedLen 是 128 的整数倍
  C3 第二轮 cached_tokens > 0（复用真的发生），且明显大于 0
"""
import json, os, subprocess, sys, time, urllib.request
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/models/Qwen3.8-27B-QUASAR-NVFP4")
KEY="maoyufeng1985"; URL="http://127.0.0.1:8080/v1/chat/completions"
UNIT=("Kubernetes schedules pods onto nodes. The scheduler filters and scores nodes, then binds the pod. "
      "Eviction removes pods when a node runs out of memory. The kubelet syncs pod state every ten seconds. ")
Q="\n\n用一句话说明 pod 是怎么被调度到节点上的。"
EXTRA=("The apiserver validates the pod spec, persists it to etcd, and notifies watchers. "
       "Controllers reconcile the desired state until the observed state matches. ")*60
def call(msgs, max_tokens=8):
    body=json.dumps({"model":"Qwen3.8-27B","messages":msgs,"max_tokens":max_tokens,"temperature":0}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as r: d=json.load(r)
    det=(d.get("usage",{}).get("prompt_tokens_details") or {})
    return d["usage"]["prompt_tokens"], det.get("cached_tokens",0), time.time()-t0, (d["choices"][0]["message"].get("content") or "")
def trace(t0):
    out=subprocess.run(["journalctl","-u","ftllm-server.service","--no-pager","--since",f"@{t0:.0f}"],
                       capture_output=True,text=True).stdout
    return [l.split("]: ",1)[-1].strip() for l in out.splitlines() if "prefixcache" in l]
# 造一条总长故意不是 128 整数倍的长提示词
N=380
P=UNIT*N+Q+EXTRA[:1400]
print("=== 第一轮（冷）===")
t=time.time(); pt1,ct1,dt1,ans=call([{"role":"user","content":P}]); L1=trace(t)
print(f"  prompt={pt1}  非对齐余数={pt1%128}  cached={ct1}  {dt1:.2f}s")
for x in L1: print("   ", x[:112])
print()
print("=== 第二轮（前缀 = 第一轮 + 追加问答）===")
time.sleep(3)
t=time.time(); pt2,ct2,dt2,_=call([{"role":"user","content":P},{"role":"assistant","content":ans},{"role":"user","content":"继续"}])
L2=trace(t)
print(f"  prompt={pt2}  cached={ct2}  {dt2:.2f}s")
for x in L2: print("   ", x[:112])
print()
ok1 = any("record OK" in x for x in L1)
ok2 = ct2 > 0
print(f"C1 两轮成功: {'是' if pt1>0 and pt2>0 else '否'}")
print(f"C2 第一轮记下快照: {'是' if ok1 else '否'}")
print(f"C3 第二轮复用: {'是，cached_tokens=%d（%.1f%%）' % (ct2, 100.0*ct2/pt2) if ok2 else '否'}")
sys.exit(0 if (ok1 and ok2) else 1)
