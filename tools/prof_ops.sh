#!/bin/bash
# prof_ops.sh -- 用 nsys 抓一次 8K 基准的执行轨迹，产出"每个算子占多少"的原始数据。
#
# 目的：回答"V100 上优化空间在哪"。要拿的是**内核自己的时长**（不是墙钟，
# 也不是被 cudaStreamSynchronize 排空污染的耗时——这个会话在这上面错过两次）。
#
# 产物：
#   /home/nsys/ops8k.sqlite      nsys 数据库（约 50-70 MB）
#   /tmp/ops8k_summary.txt       内核按时长排序的汇总（只看这个，别 cat 原始日志）
#
# 用法：XID_BASE=413 tools/prof_ops.sh
set -u
cd /home/fastllm
XID_BASE="${XID_BASE:-0}"
OUT=/home/nsys/ops8k
SUM=/tmp/ops8k_summary.txt
export PYTHONPATH=build-sm70-tests/tools

: > "$SUM"
echo "start=$(date '+%F %T')  Xid基线=$XID_BASE  卡数=$(timeout 10 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)" >> "$SUM"

# nsys 包住整条命令；外面再套看门狗，这样卡死时能抓现场
tools/gpu_watchdog.sh /tmp/ops8k.out --timeout 400 --label ops8k -- \
  env NSYS_OUTPUT_DIR=/home/nsys \
  nsys profile --force-overwrite=true -o "$OUT" \
    --trace=cuda,nvtx --cuda-memory-usage=false --stats=false \
  env FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 \
  python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
    --tp 4 --dtype auto --enable_thinking false --prefix_cache false \
    --temperature 0 --top_k 1 --tokens 200000 --max_batch 1 --gpu_mem_ratio 0.98 \
    --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size 4096 \
    --input_tokens 8192 --output_tokens 8 --batch 1 --warmup 0 > /tmp/ops8k_bench.log 2>&1
echo "nsys rc=$?" >> "$SUM"

echo "--- 基准结果行 ---" >> "$SUM"
grep -aE "Total time|sha256" /tmp/ops8k_bench.log 2>/dev/null | tail -2 >> "$SUM"

echo "--- Xid: 跑前=$XID_BASE 跑后=$(timeout 15 dmesg 2>/dev/null | grep -cE 'NVRM: Xid') ---" >> "$SUM"

if [ -f "$OUT.sqlite" ]; then
  echo "--- 内核按时长排序（前 20）---" >> "$SUM"
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      "$OUT.sqlite" 2>/dev/null | head -24 >> "$SUM"
  echo "--- 内核总时长 ---" >> "$SUM"
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      "$OUT.sqlite" 2>/dev/null | tail -3 >> "$SUM"
else
  echo "!! 没有产出 $OUT.sqlite" >> "$SUM"
fi
echo "PROFDONE" >> "$SUM"
