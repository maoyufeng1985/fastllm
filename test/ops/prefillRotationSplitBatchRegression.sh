#!/usr/bin/env bash
# Regression for the host-side segfault in the split batch forward path.
#
# When two long prompts are both in the middle of a chunked prefill, the
# scheduler latches isIntermediateChunkedPrefill, the forward returns no token
# by design, and runSplitBatchForward used to read index 0 of that empty
# vector. The crash only happens once two prefills are genuinely in flight, so
# the test asserts that the shape was actually reached (inFlight=2) rather than
# merely that the process happened to survive.
#
# Needs 4x V100-16GB, the QUASAR-NVFP4 checkpoint, and a built libfastllm_tools.
set -u

REPO=${REPO:-/home/fastllm}
BUILD=${BUILD:-$REPO/build-sm70-tests}
MODEL=${MODEL:-/home/models/Qwen3.8-27B-QUASAR-NVFP4}
TOKENS=${TOKENS:-65536}
PROMPT=${PROMPT:-20480}
OUT=${OUT:-32}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model not present at $MODEL"
    exit 77
fi
if [ ! -f "$BUILD/tools/ftllm/libfastllm_tools.so" ]; then
    echo "SKIP: $BUILD/tools/ftllm/libfastllm_tools.so not built"
    exit 77
fi

# The box is shared; a neighbour on the GPUs turns this into an OOM instead of
# a rotation test, so refuse to run rather than report a false failure.
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
       awk '$1 > 1024 { n++ } END { print n + 0 }')
if [ "$busy" != "0" ]; then
    echo "SKIP: $busy GPU(s) still holding >1 GiB; wait for the box to be idle"
    exit 77
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ROTATE is overridable so the guard itself can be exercised: with it off the
# two prefills stay serial, inFlight never reaches 2, and the test must fail
# loudly instead of passing vacuously.
PYTHONPATH="$BUILD/tools" \
FASTLLM_PREFILL_ROTATE=${ROTATE:-1} \
FASTLLM_SCHED_TRACE=1 \
python3 -m ftllm.cli benchmark "$MODEL" \
    --tp 4 --cuda_embedding --max_batch 4 --tokens "$TOKENS" --dtype auto \
    --enable_thinking false --prefix_cache false --input_tokens "$PROMPT" \
    --output_tokens "$OUT" --batch 2 --warmup 0 --temperature 0 --top_k 1 \
    > "$WORK/out" 2> "$WORK/err"
status=$?

# A signal death (139 = SIGSEGV) is the failure this test exists for.
if [ "$status" -ne 0 ]; then
    echo "FAIL: benchmark exited $status (139 means the split batch path crashed)"
    tail -5 "$WORK/err"
    exit 1
fi

if ! grep -q '^Summary' "$WORK/out"; then
    echo "FAIL: no benchmark summary; the run did not complete"
    tail -5 "$WORK/out"
    exit 1
fi

if ! grep -q 'inFlight=2' "$WORK/err"; then
    echo "FAIL: two prefills never overlapped (no inFlight=2), so this run did"
    echo "      not exercise the path the regression covers."
    exit 1
fi

# The yield protocol has to stay intact: an intermediate chunk produces no
# token, and the run must still report its full output for both requests.
outputs=$(grep -oE 'Actual output tokens +[0-9]+' "$WORK/out" | grep -oE '[0-9]+$')
expected=$((OUT * 2))
if [ "$outputs" != "$expected" ]; then
    echo "FAIL: produced ${outputs:-none} output tokens, expected $expected"
    exit 1
fi

echo "PASS: prefillRotationSplitBatchRegression ($(grep -c 'inFlight=2' "$WORK/err") overlapping rounds, $outputs tokens)"
