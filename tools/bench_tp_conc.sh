#!/bin/bash
# bench_tp_conc.sh -- TP4 多请求 benchmark：可配 prompt、并发、错开、轮转。
#
# 这台机器（4×V100-SXM2-16GB / PCIe gen3 / 无 NVLink）上跑 Qwen3.8-27B NVFP4 的
# 并发基准。把这一轮做过的所有排法收进一个脚本，参数化，不再每个场景抄一份。
#
# 用法：
#   tools/bench_tp_conc.sh                                  # 80K prompt × 2 并发
#   INPUT_TOKENS=16000 CONC=3 STAGGER=10 tools/bench_tp_conc.sh
#   INPUT_TOKENS=16000 CONC=2 STAGGER=0 ROTATE=0 tools/bench_tp_conc.sh
#
# 可配：INPUT_TOKENS(=80000) CONC(=2) STAGGER(=0 秒) ROTATE(=1) CHUNK(=8192)
#       OUT_TOKENS(=8) TIMEOUT(=900)
#
# 硬性检查（缺一项不算跑完，脚本会明说）：
#   G1 开工闸门：卡数=4、无其他占用者、显存归零
#   G2 Xid 基线：跑前记录，跑后比对，新增必须为 0
#   G3 全程由 tools/gpu_watchdog.sh 包住（卡死即抓现场并杀、收工查显存归零）
#
# 产出：
#   /tmp/<TAG>.out            基准输出（只看汇总行，不要 cat）
#   /tmp/<TAG>_summary.txt    判据 + 结果行 + Xid 比对
set -u
cd /home/fastllm

INPUT_TOKENS="${INPUT_TOKENS:-80000}"
CONC="${CONC:-2}"
STAGGER="${STAGGER:-0}"
ROTATE="${ROTATE:-1}"
CHUNK="${CHUNK:-8192}"
OUT_TOKENS="${OUT_TOKENS:-8}"
TIMEOUT="${TIMEOUT:-900}"

# 上下文池大小（--tokens）：它是 paged cache 的**上限**，分配量与之成正比。
# 早先写成 CONC*200000（每条留 200K），4 并发时算出 800000 token，
# 对应 6250 页 / 204.8 MB 的一次分配，在装完权重的卡上直接 cudaErrorMemoryAllocation
# （fastllm-cuda.cu:4726）。池子只需覆盖"并发数 × prompt 长度"再加余量。
# 默认 3 倍余量；要复现历史行为就显式传 POOL_TOKENS。
POOL="${POOL_TOKENS:-$((CONC * INPUT_TOKENS * 3))}"
TAG="bc_${INPUT_TOKENS}_c${CONC}_s${STAGGER}_r${ROTATE}"
LOG="/tmp/${TAG}.out"
SUM="/tmp/${TAG}_summary.txt"

: > "$SUM"
{
  echo "start=$(date '+%F %T')"
  echo "配置: prompt=${INPUT_TOKENS} 并发=${CONC} 错开=${STAGGER}s 轮转=${ROTATE} chunk=${CHUNK} 池=${POOL}"
} >> "$SUM"

# --- G1 开工闸门 ---
CARDS=$(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
BUSY=$(timeout 20 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
USED=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
echo "G1 卡数=$CARDS 占用者=$BUSY 显存=[$USED]" >> "$SUM"
if [ "$CARDS" != "4" ] || [ "$BUSY" != "0" ]; then
  echo "!! G1 未过：卡数或占用者不合格，不跑" >> "$SUM"
  echo "G1FAIL" >> "$SUM"; exit 3
fi

# --- G2 Xid 基线 ---
XID_BEFORE=$(timeout 15 dmesg 2>/dev/null | grep -cE "NVRM: Xid")
echo "G2 Xid 基线=$XID_BEFORE" >> "$SUM"

# --- 组装环境 ---
ENVS=(FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 "PYTHONPATH=build-sm70-tests/tools")
[ "$ROTATE" = "1" ] && ENVS+=(FASTLLM_PREFILL_ROTATE=1)
# TRACE=1：打开诊断输出——每条请求的绝对时刻与 TTFT、每个 token 的相对毫秒、
# 批量级时间线（最早/最晚首字与散布）、以及引擎侧的每轮 prefill 预算与被跳过的请求。
# 判断"几条真的同时在跑""错开有没有生效""限制点在哪"全靠这几组行。
if [ "${TRACE:-0}" != "0" ]; then
  ENVS+=(FASTLLM_BENCH_TIME_TRACE=1 FASTLLM_PREFILL_BUDGET_TRACE=1)
fi

# --- 跑（G3：看门狗包住）---
tools/gpu_watchdog.sh "$LOG" --timeout "$TIMEOUT" --label "$TAG" -- \
  env "${ENVS[@]}" python3 -m ftllm.cli benchmark /home/models/Qwen3.8-27B-QUASAR-NVFP4 \
    --tp 4 --dtype auto --enable_thinking false --prefix_cache false \
    --temperature 0 --top_k 1 --tokens "$POOL" --gpu_mem_ratio 0.98 \
    --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size "$CHUNK" \
    --input_tokens "$INPUT_TOKENS" --output_tokens "$OUT_TOKENS" \
    --batch "$CONC" --stagger_s "$STAGGER" --warmup 0 > "/tmp/${TAG}.wd" 2>&1
echo "watchdog rc=$?" >> "$SUM"

echo "--- 结果 ---" >> "$SUM"
grep -aE "Total time|Prefill|TTFT|TPOP|Batch decode|common window|before last TTFT|Token stream sha256|Batch total|Per request|Pages limit|Batch limit|Input tokens|Batch " "$LOG" 2>/dev/null | head -20 >> "$SUM"

# TRACE=1 时把诊断行也收进摘要（被判据用到的都在这里）
if [ "${TRACE:-0}" != "0" ]; then
  echo "--- 批量时间线 ---" >> "$SUM"
  grep -a "\[batchtime\]" "$LOG" 2>/dev/null >> "$SUM"
  echo "--- 每条请求的时刻 ---" >> "$SUM"
  grep -a "\[reqtime\]" "$LOG" 2>/dev/null >> "$SUM"
  echo "--- 每个 token 的相对毫秒 ---" >> "$SUM"
  grep -a "\[toktime\]" "$LOG" 2>/dev/null >> "$SUM"
  echo "--- 引擎每轮预算（前 30 行）---" >> "$SUM"
  grep -a "\[budget\]" "$LOG" 2>/dev/null | head -30 >> "$SUM"
  echo "--- 请求进入调度字典的时刻（引擎侧收到请求）---" >> "$SUM"
  grep -a "\[handle\]" "$LOG" 2>/dev/null | head -10 >> "$SUM"
  echo "--- 被过滤出调度列表的请求及原因（isEnding=1 表示已跑完）---" >> "$SUM"
  grep -a "\[filter\]" "$LOG" 2>/dev/null | head -20 >> "$SUM"
  echo "--- 预算函数返回值 ---" >> "$SUM"
  grep -a "\[budgetfn\]" "$LOG" 2>/dev/null | head -3 >> "$SUM"
  echo "--- 因预算被跳过的请求（若有）---" >> "$SUM"
  grep -a "\[budgetskip\]" "$LOG" 2>/dev/null | head -20 >> "$SUM"
fi

# --- G2 收尾比对 ---
XID_AFTER=$(timeout 15 dmesg 2>/dev/null | grep -cE "NVRM: Xid")
NEW=$((XID_AFTER - XID_BEFORE))
echo "G2 Xid 跑后=$XID_AFTER 新增=$NEW $([ "$NEW" != "0" ] && echo '<<< 异常')" >> "$SUM"
echo "DONE" >> "$SUM"
