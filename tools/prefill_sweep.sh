#!/bin/bash
# prefill_sweep.sh -- prefill 的两个杠杆测量：有效 chunk 大小、KV dtype。
#
# 代码依据（为什么 chunk 不是 8192）：
#   开前缀缓存时 Qwen3_5Model::GetChunkedPrefillSize() 返回 min(base, interval)
#   （src/models/qwen3_5.cpp:10167-10177）；
#   interval = FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES(默认16) × pageLen(128) = 2048
#   （src/models/qwen3_5.cpp:4790-4794；pageLen 默认 128 见 src/fastllm.cpp:295）。
#   所以 --chunked_prefill_size 8192 被 2048 盖住。
#
# 本脚本用 --prefix_cache_snapshot_interval_pages 改**有效 chunk**，其余与服务一致：
#   pages=16 -> 2048（今天）   pages=32 -> 4096   pages=64 -> 8192
#
# 每轮自验（AGENTS.md）：
#   C1 有 PROBE_RESULT        C2 sha256 与参照一致（fp8 各轮之间）
#   C3 日志里 `Long prefill chunk: ... size=N` 等于期望值（证明分块真的变了）
#
# 每轮用 tools/gpu_watchdog.sh 包住。只 grep 汇总行，不 cat 原始日志。
# 用法:  DRYRUN=1 tools/prefill_sweep.sh   |   tools/prefill_sweep.sh
set -u
cd /home/fastllm
export PYTHONPATH=/home/fastllm/build-sm70-tests/tools

OUT=/tmp/pcab
mkdir -p "$OUT"
MODEL=/home/models/Qwen3.8-27B-QUASAR-NVFP4

COMMON=(--tp 4 --dtype auto --low_gpu_mem --gpu_mem_ratio 0.98 --max_batch 8
        --enable_thinking false --temperature 0 --top_k 1 --tokens 200000
        --chunked_prefill_size 8192 --prefix_cache true --batch 1 --warmup 0)

# name|input_tokens|output_tokens|interval_pages|kv_dtype
RUNS="${RUNS:-i16_c2048_fp8|16384|8|16|fp8_e4m3
i16_c4096_fp8|16384|8|32|fp8_e4m3
i16_c8192_fp8|16384|8|64|fp8_e4m3
i16_c2048_fp4|16384|8|16|fp4
i16_c8192_fp4|16384|8|64|fp4
i8_c2048_fp8|8192|8|16|fp8_e4m3
i8_c8192_fp8|8192|8|64|fp8_e4m3}"

ALL="$OUT/ALL.sum"
echo "# start=$(date '+%F %T')  model=$MODEL" >> "$ALL"

while IFS='|' read -r name in_tok out_tok pages kvd; do
  [ -z "$name" ] && continue
  log="$OUT/$name.out"
  sum="$OUT/$name.sum"

  cmd=(python3 tools/prefill_probe.py "$MODEL" "${COMMON[@]}"
       --input_tokens "$in_tok" --output_tokens "$out_tok"
       --kv_cache_dtype "$kvd"
       --prefix_cache_snapshot_interval_pages "$pages")

  if [ "${DRYRUN:-0}" != "0" ]; then
    echo "[DRYRUN] $name expect_chunk=$((pages*128)): ${cmd[*]}"
    continue
  fi

  echo "=== $name start=$(date '+%T') expect_chunk=$((pages*128)) ==="
  tools/gpu_watchdog.sh "$log" --timeout 900 --label "$name" -- \
    "${cmd[@]}" > "$log.wd" 2>&1
  rc=$?

  {
    echo "### $name in=$in_tok pages=$pages expect_chunk=$((pages*128)) kv=$kvd rc=$rc $(date '+%F %T')"
    grep -aE "PROBE input_tokens|PROBE_RESULT|PROBE_SHA|PROBE_SAMPLE" "$log" 2>/dev/null
    grep -aE "Long prefill chunk" "$log" 2>/dev/null | head -1
    echo -n "prompt_lines_chunks: "
    grep -acE "^\[Prompt\] " "$log" 2>/dev/null
    grep -aE "^\[Prompt\] " "$log" 2>/dev/null | head -20 | sed 's/^/  /'
    grep -aE "GPU_WATCHDOG verdict" "$log.wd" 2>/dev/null | head -1
  } > "$sum"
  cat "$sum" >> "$ALL"
  echo
done <<< "$RUNS"

echo "ALL_DONE $(date '+%F %T')" >> "$ALL"
