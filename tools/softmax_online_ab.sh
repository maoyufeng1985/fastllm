#!/usr/bin/env bash
# A/B harness for the prefill causal-softmax online variant.
#
# Runs the same 180K prefill twice, once on the production softmax structure and
# once with FASTLLM_PAGED_SOFTMAX_ONLINE=1, then prints the pair of walls, the
# prefill tok/s, and the greedy token-stream hash of each run. The hash is the
# project's correctness gate (see docs/sm70_status_and_backlog.md): the two runs
# must agree, because the variant only changes how the softmax row is traversed,
# not what it computes.
#
#   tools/softmax_online_ab.sh            # both arms, 180K
#   INPUT_TOKENS=32000 tools/softmax_online_ab.sh --quick
#
# Refuses to start while another process holds the GPUs, because a collision
# silently corrupts both the timing and the token hash.
set -u

REPO=/home/fastllm
MODEL=/home/models/Qwen3.8-27B-QUASAR-NVFP4
INPUT_TOKENS=${INPUT_TOKENS:-180000}
TOKENS_POOL=${TOKENS_POOL:-200000}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-8}
QUICK=${1:-}
if [ "$QUICK" = "--quick" ]; then
  INPUT_TOKENS=${INPUT_TOKENS:-32000}
  TOKENS_POOL=${TOKENS_POOL:-40000}
fi

OUTDIR=${OUTDIR:-/tmp/softmax-online-ab}
mkdir -p "$OUTDIR"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "ABORT: $busy GPU process(es) already running; timing and hash would be unreliable:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
  exit 2
fi

run_arm() {
  local label=$1 switch=$2
  local log="$OUTDIR/${label}.log"
  echo "=== arm: $label (FASTLLM_PAGED_SOFTMAX_ONLINE=$switch) input=$INPUT_TOKENS ==="
  ( cd "$REPO" && env PYTHONPATH=build-sm70-tests/tools \
      FASTLLM_PAGED_SOFTMAX_ONLINE="$switch" \
      timeout 1800 python3 -m ftllm.cli benchmark "$MODEL" \
        --tp 4 --dtype auto --enable_thinking false --prefix_cache false \
        --temperature 0 --top_k 1 --tokens "$TOKENS_POOL" --max_batch 1 \
        --gpu_mem_ratio 0.98 --kv_cache_dtype fp8_e4m3 --low_gpu_mem \
        --input_tokens "$INPUT_TOKENS" --output_tokens "$OUTPUT_TOKENS" \
        --batch 1 --warmup 0 --chunked_prefill_size 4096 > "$log" 2>&1 )
  local rc=$?
  local hash
  # the benchmark prints: Token stream sha256    <hex> (tokens/request: N)
  hash=$(grep -oE 'Token stream sha256 +[0-9a-f]+' "$log" | tail -1 | grep -oE '[0-9a-f]{16,}')
  local prefill
  prefill=$(grep 'Prefill' "$log" | grep -oE '[0-9.]+ tokens/s' | head -1)
  local wall
  wall=$(grep -oE 'Total time +[0-9.]+ s' "$log" | tail -1)
  echo "  rc=$rc token_hash=${hash:-<none>} prefill=${prefill:-<none>} ${wall:-}"
  echo "  log=$log"
  printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "${hash:-none}" "${prefill:-none}" >> "$OUTDIR/summary.tsv"
}

: > "$OUTDIR/summary.tsv"
run_arm prod 0
run_arm online 1

echo
echo "=== summary (label rc token_hash prefill) ==="
cat "$OUTDIR/summary.tsv"
h1=$(awk -F'\t' '$1=="prod"{print $3}' "$OUTDIR/summary.tsv")
h2=$(awk -F'\t' '$1=="online"{print $3}' "$OUTDIR/summary.tsv")
if [ "$h1" = "none" ] || [ "$h2" = "none" ]; then
  echo "VERDICT: inconclusive, missing token hash on at least one arm"
elif [ "$h1" = "$h2" ]; then
  echo "VERDICT: token hash matches ($h1) -> variant is numerically equivalent on this workload"
else
  echo "VERDICT: HASH MISMATCH prod=$h1 online=$h2 -> do not enable the switch"
fi
