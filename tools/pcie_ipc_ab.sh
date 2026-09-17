#!/bin/bash
# pcie_ipc_ab.sh -- FASTLLM_PCIE_IPC_AR 开启/关闭 的对照测量，每轮自验。
#
# 为什么要"每轮自验"：这台机器会偶发四卡掉线（Xid 79/154，原因未确定）。
# 崩掉的那一轮会留下一个**没有 Total time**的日志，以及一个**数值可能不完整**的输出。
# 如果不自验就把它当"一次样本"用，结论会被污染。所以每一轮都必须过三条判据，
# 不通过的轮次**直接作废**，并明确标出来。
#
# 三条判据（缺一条即作废）：
#   C1 跑完   ：日志里出现 "Total time"
#   C2 数值对 ：token sha256 与 OFF 那次一致（OFF 的哈希是数值上的参照）
#   C3 真执行 ：ON 那次必须打印 "pcie_ipc AR: <n> collective(s) served" 且 n > 0
#               —— 否则"开了开关"和"没开"可能构造上相同，对照无意义
#
# 用法：
#   ORDER=on tools/pcie_ipc_ab.sh      # 先跑开启那次，再跑关闭那次（反序，解顺序混淆）
#   ORDER=off tools/pcie_ipc_ab.sh     # 先关后开（我前几轮用的顺序）
#   PAIRS=2 ORDER=on tools/pcie_ipc_ab.sh
#
# 产物：/tmp/AB_<order>_<n>_<arm>.{out,wd}，摘要打到 stdout。
set -u
cd /home/fastllm
export PYTHONPATH=build-sm70-tests/tools
ORDER="${ORDER:-on}"
PAIRS="${PAIRS:-1}"
REF_SHA="adcb1bd7"          # OFF 那次的哈希，作为数值参照
COMMON="/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto --enable_thinking false \
--prefix_cache false --temperature 0 --top_k 1 --tokens 200000 --max_batch 1 \
--gpu_mem_ratio 0.98 --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size 4096 \
--input_tokens 8192 --output_tokens 8 --batch 1 --warmup 0"

cards() { timeout 10 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l; }
xids()  { timeout 10 dmesg 2>/dev/null | grep -cE "NVRM: Xid"; }

run_arm() {   # $1=arm(on/off)  $2=轮次
  local arm="$1"
  local n="$2"
  local log="/tmp/AB_${ORDER}_${n}_${arm}.out"
  local rc
  local -a envs=(FASTLLM_QWEN35_SM70_CUDA_GRAPH=0)
  [ "$arm" = "on" ] && envs+=(FASTLLM_PCIE_IPC_AR=1)
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "    [dryrun] 会跑: 开关=$arm 日志=$log 环境=${envs[*]}"
    return 0
  fi
  timeout 300 tools/gpu_watchdog.sh "$log" --timeout 220 --label "${ORDER}${n}${arm}" -- \
      env "${envs[@]}" python3 -m ftllm.cli benchmark $COMMON \
      > "/tmp/AB_${ORDER}_${n}_${arm}.wd" 2>&1
  rc=$?

  if [ "${DRYRUN:-0}" = "1" ]; then echo "  run$n-$arm [dryrun]"; return 0; fi
  local total sha served verdict ok="OK" why=""
  total=$(grep -a "Total time" "$log" 2>/dev/null | tail -1 | awk '{print $3}')
  sha=$(grep -a "Token stream sha256" "$log" 2>/dev/null | tail -1 | awk '{print $4}')
  served=$(grep -a "pcie_ipc AR:" "$log" 2>/dev/null | tail -1 | sed 's/.*AR: //;s/ collective.*//')
  verdict=$(grep -a GPU_WATCHDOG "/tmp/AB_${ORDER}_${n}_${arm}.wd" 2>/dev/null | head -1 | sed 's/.*verdict=/verdict=/;s/ elapsed.*//')

  [ -z "$total" ] && { ok="VOID"; why="C1未过(无Total time)"; }
  if [ -n "$sha" ] && [ "${sha:0:8}" != "$REF_SHA" ]; then ok="VOID"; why="$why C2未过(哈希=${sha:0:8})"; fi
  if [ "$arm" = "on" ]; then
    if [ -z "$served" ] || [ "$served" = "0" ]; then ok="VOID"; why="$why C3未过(served=${served:-无})"; fi
  fi
  # 掉卡立刻停
  local c; c=$(cards)
  if [ "$c" != "4" ]; then ok="VOID"; why="$why 掉卡(卡数=$c)"; fi

  echo "  run$n-$arm  rc=$rc Total=${total:-无} sha=${sha:0:8} served=${served:-0} $verdict  自验=$ok $why"
  [ "$ok" = "VOID" ] && return 90
  return 0
}

echo "=== 臂序=$ORDER  轮数=$PAIRS  参照哈希=$REF_SHA  开工前卡数=$(cards) Xid=$(xids) ==="
for i in $(seq 1 "$PAIRS"); do
  if [ "$ORDER" = "on" ]; then
    run_arm on  "$i"; rc1=$?
    [ "$rc1" = "90" ] && { echo ">>> 轮$i 作废（或掉卡），停手"; break; }
    run_arm off "$i"; rc2=$?
    [ "$rc2" = "90" ] && { echo ">>> 轮$i 作废（或掉卡），停手"; break; }
  else
    run_arm off "$i"; rc1=$?
    [ "$rc1" = "90" ] && { echo ">>> 轮$i 作废（或掉卡），停手"; break; }
    run_arm on  "$i"; rc2=$?
    [ "$rc2" = "90" ] && { echo ">>> 轮$i 作废（或掉卡），停手"; break; }
  fi
done
echo "ABDONE"
