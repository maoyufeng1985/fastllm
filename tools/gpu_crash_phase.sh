#!/bin/bash
# gpu_crash_phase.sh -- 同一开机内连跑 N 次普通基准，逐次记录"在哪个阶段崩的"。
#
# 背景：这台机器（4xV100 / PCIe）多次四卡同时掉总线
# （Xid 79 + Xid 154，pcieport 0000:04:04.0 报 Multiple Uncorrectable，
# 四卡 AER: can't recover）。掉卡无法恢复，只能重启。
#
# 已经排掉的解释（都有实测反证）：
#   - "和开机后多少秒有关"：开机后 93 秒起跑跑完了，111 秒起跑崩了 -> 否掉
#   - "和 pcie_ipc / push4 开关有关"：关掉开关也崩过 -> 否掉
#   - "和连跑第几次有关"：有跑满 3 次不崩的，也有第 1 次就崩的 -> 单独不成立
# 所以本脚本测的是**同一开机里累积跑，看跑到第几次、在哪个阶段崩**，
# 并逐秒记录当时的利用率/显存/温度/功耗，把"崩点"和"当时在做什么"对上。
#
# 用法：
#   tools/gpu_crash_phase.sh                        # 跑 1 次
#   LABEL=a RUNS=5 tools/gpu_crash_phase.sh         # 同一开机内连跑 5 次
#
# 产出（每次一套）：
#   /tmp/PHASE_<label>_<n>.out       基准输出
#   /tmp/PHASE_<label>_<n>.samples   每秒一行：开机后秒数 | 卡数 | 逐卡 util/显存/温度/功耗 | 阶段
#   /tmp/PHASE_<label>_<n>.summary   结论 + 崩点阶段 + 首个错误行 + 崩溃前 5 个采样
#
# 阶段（只看摘要行，避免把进度条读进来）：
#   LOAD 加载/预热中 | SETUP 加载完做准备 | RUNNING 进入计算 | DONE 跑完 | ERROR 报错
# 崩了立刻停，不继续跑下一次（设备已经没了，继续没意义）。
set -u
cd /home/fastllm
export PYTHONPATH=build-sm70-tests/tools
LABEL="${LABEL:-run}"
RUNS="${RUNS:-1}"
BOOT=$(date -d "$(uptime -s)" +%s)

COMMON="/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto --enable_thinking false \
--prefix_cache false --temperature 0 --top_k 1 --tokens 200000 --max_batch 1 \
--gpu_mem_ratio 0.98 --kv_cache_dtype fp8_e4m3 --low_gpu_mem --chunked_prefill_size 4096 \
--input_tokens 8192 --output_tokens 8 --batch 1 --warmup 0"

LOG=""; SAMP=""; SUM=""

phase() {
  if grep -qa "Total time" "$LOG" 2>/dev/null; then echo DONE; return; fi
  if grep -qaE "RuntimeError|CUDA error|cublas error" "$LOG" 2>/dev/null; then echo ERROR; return; fi
  if grep -qa "Long prefill chunk" "$LOG" 2>/dev/null; then echo RUNNING; return; fi
  if grep -qa "finish\." "$LOG" 2>/dev/null; then echo SETUP; return; fi
  echo LOAD
}

errline() {
  grep -aoE "(RuntimeError|CUDA error|cublas error)[^\"]{0,60}" "$LOG" 2>/dev/null | head -1
}

cards() {
  timeout 10 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
}

run_once() {
  local N="$1"
  LOG="/tmp/PHASE_${LABEL}_$N.out"
  SAMP="/tmp/PHASE_${LABEL}_$N.samples"
  SUM="/tmp/PHASE_${LABEL}_$N.summary"
  : > "$SAMP"
  {
    echo "boot=$(date -d @$BOOT '+%F %T')  label=$LABEL  run=$N/$RUNS"
    echo "起跑=开机后$(($(date +%s)-BOOT))s  卡数=$(cards)"
  } > "$SUM"

  timeout 300 tools/gpu_watchdog.sh "$LOG" --timeout 240 --label "phase_${LABEL}_$N" -- \
    env FASTLLM_QWEN35_SM70_CUDA_GRAPH=0 \
    python3 -m ftllm.cli benchmark $COMMON > "/tmp/PHASE_${LABEL}_$N.wd" 2>&1 &
  local WDPID=$!

  local last=LOAD
  while kill -0 $WDPID 2>/dev/null; do
    sleep 1
    local row n ph
    row=$(timeout 8 nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
          --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
    n=$(echo "$row" | wc -w)
    ph=$(phase); last=$ph
    echo "$(($(date +%s)-BOOT))s | cards=$((n/4)) | ${row:-none} | $ph" >> "$SAMP"
    if [ -z "$row" ] || [ "$n" = "0" ]; then
      local xid
      xid=$(timeout 15 dmesg 2>/dev/null | grep -cE "NVRM: Xid")
      {
        echo "结论: CRASHED (第 $N 次)"
        echo "崩溃时: 开机后 $(($(date +%s)-BOOT))s   阶段=$ph"
        echo "Xid 计数=$xid   首个错误行: $(errline)"
        echo "--- 崩溃前最后 5 个采样 ---"
        tail -5 "$SAMP"
        echo "--- dmesg 尾部 ---"
        timeout 15 dmesg 2>/dev/null | grep -E "NVRM: Xid|AER|pcieport" | tail -6
      } >> "$SUM"
      echo "RUN$N CRASH 开机后$(($(date +%s)-BOOT))s 阶段=$ph  $(errline)"
      return 90
    fi
  done

  wait $WDPID 2>/dev/null; local rc=$?
  {
    echo "结论: 结束（未掉线） rc=$rc (第 $N 次)"
    echo "结束于: 开机后 $(($(date +%s)-BOOT))s   最后阶段=$last"
    echo "Total=$(grep -a 'Total time' "$LOG" 2>/dev/null | tail -1 | awk '{print $3}')  sha=$(grep -a 'Token stream sha256' "$LOG" 2>/dev/null | tail -1 | awk '{print $4}' | cut -c1-16)"
    echo "首个错误行: $(errline)"
    echo "--- 最后 3 个采样 ---"
    tail -3 "$SAMP"
  } >> "$SUM"
  echo "RUN$N OK   rc=$rc 结束于开机后$(($(date +%s)-BOOT))s 阶段=$last $(errline)"
  return 0
}

for i in $(seq 1 "$RUNS"); do
  if [ "$(cards)" != "4" ]; then
    echo "RUN$i ABORT: 开工前只剩 $(cards) 张卡"
    break
  fi
  run_once "$i"
  rc=$?
  if [ "$rc" = "90" ]; then
    echo ">>> 第 $i 次崩了，停手（等重启）"
    break
  fi
done
echo "PHASEDONE"
