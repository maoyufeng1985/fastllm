#!/bin/bash
# push4_ab.sh -- FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4 开关各跑 N 次的对照测量。
#
# 为什么要重测（2026-09-17 那次只跑了一次，而且两个变量一起动了）：
#   关闭那次跑的是 auto，日志显示只启用了 6 条精度/尺寸路径里的 3 条；
#   打开那次跑的是 forced（FASTLLM_CUDA_CUSTOM_ALLREDUCE=1），6 条全开、
#   并且关掉了 ">= 40 KiB 交给 NCCL" 那条规则。所以那次的差不能干净地归给 push4。
#   本脚本两个开关组都设 FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 与 FASTLLM_CUSTOM_AR_CENSUS=1，
#   唯一的差别是 PUSH4 本身。
#
# 四条判据（缺一条即作废）：
#   C1 跑完   ：日志里出现 "Total time"
#   C2 数值对 ：token sha256 与参照一致（bench 档是 d5fcc5fc，换档取第一次 off 跑）
#   C3 真执行 ：PUSH4=1 的那几次，"launched on the TP4 push kernel" 次数 > 0；
#               PUSH4=0 的那几次，同一行次数必须为 0（反向检查，防"两边都跑了新代码"）
#   C4 真吐字 ：TPOP 必须是数字。32K 输入那次吐了 0 个 token，Total time 和哈希都在，
#               只有前三条会判成 OK——那是个假通过，所以补这一条。
#
# 用法：
#   tools/push4_ab.sh                  # 真跑，顺序 off on on off off on
#   DRYRUN=1 tools/push4_ab.sh         # 只打印将要执行的命令行，不碰显卡
#   PROFILE=prod tools/push4_ab.sh     # 换成生产服务的旗标组合再跑一遍
#   SEQ="on off" tools/push4_ab.sh     # 换顺序 / 换轮数
#
# PROFILE 两档：
#   bench（缺省）= 历史对照那套旗标（--cuda_embedding --max_batch 1 --tokens 16384）
#   prod        = 生产服务那套旗标（--low_gpu_mem --prefix_cache true --max_batch 8 等）
#   prod32k     = prod 旗标 + 输入 32K（生产那条会话的上下文约 28K）
# 换档时 REF_SHA 传 auto，参照哈希取第一次完成的 off 跑，只保证六次互相一致。
#
# 产物：/tmp/push4_<profile>_<n>_<arm>.out（benchmark 输出）、同名 .wd（看门狗输出）
set -u
cd /home/fastllm
export PYTHONPATH=build-sm70-tests/tools

SEQ="${SEQ:-off on on off off on}"
PROFILE="${PROFILE:-bench}"
LIB="build-sm70-tests/tools/ftllm/libfastllm_tools.so"
case "$PROFILE" in
  bench)
    REF_SHA="d5fcc5fc"
    COMMON=(/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --cuda_embedding --max_batch 1
            --tokens 16384 --dtype auto --enable_thinking false --prefix_cache false
            --input_tokens 8192 --output_tokens 128 --batch 1 --warmup 1
            --temperature 0 --top_k 1)
    ;;
  prod)
    REF_SHA="auto"
    COMMON=(/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto
            --kv_cache_dtype fp8_e4m3 --low_gpu_mem --gpu_mem_ratio 0.98
            --prefix_cache true --chunked_prefill_size 8192 --max_batch 8
            --enable_thinking false --input_tokens 8192 --output_tokens 128
            --batch 1 --warmup 1 --temperature 0 --top_k 1)
    ;;
  # prod 旗标，但输入拉到 32K：生产那条会话的上下文约 28K，8K 那组不能直接代表它。
  prod32k)
    REF_SHA="auto"
    COMMON=(/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto
            --kv_cache_dtype fp8_e4m3 --low_gpu_mem --gpu_mem_ratio 0.98
            --prefix_cache true --chunked_prefill_size 8192 --max_batch 8
            --enable_thinking false --input_tokens 32768 --output_tokens 128
            --batch 1 --warmup 1 --temperature 0 --top_k 1)
    ;;
  # 同上但输入 16K。32K 这次吐 0 个 token（见 C4），所以降一档取长上下文的点。
  prod16k)
    REF_SHA="auto"
    COMMON=(/home/models/Qwen3.8-27B-QUASAR-NVFP4 --tp 4 --dtype auto
            --kv_cache_dtype fp8_e4m3 --low_gpu_mem --gpu_mem_ratio 0.98
            --prefix_cache true --chunked_prefill_size 8192 --max_batch 8
            --enable_thinking false --input_tokens 16384 --output_tokens 128
            --batch 1 --warmup 1 --temperature 0 --top_k 1)
    ;;
  *) echo "!! 未知 PROFILE=$PROFILE（只认 bench / prod）"; exit 2 ;;
esac
COMMON_ENVS=(FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 FASTLLM_CUSTOM_AR_CENSUS=1)

cards() { timeout 10 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l; }
xids()  { timeout 10 dmesg 2>/dev/null | grep -cE "NVRM: Xid"; }

# 开局自检：跑分程序加载的是 tools/ftllm 下那块库，不是 build 根目录那块。
# 新加的计数字符串只在重新编译过那块库之后才在里面；不在就说明会跑到旧代码。
preflight() {
  if [ ! -f "$LIB" ]; then
    echo "!! 找不到 $LIB"; return 1
  fi
  local hits
  hits=$(strings -a "$LIB" 2>/dev/null | grep -c "launched on the TP4 push kernel")
  echo "== 库里新计数串命中 $hits 次（必须 >=1，否则跑的是旧库）"
  [ "$hits" -ge 1 ] || return 1
  return 0
}

run_arm() {   # $1=arm(off/on)  $2=轮次
  local arm="$1"
  local n="$2"
  local log="/tmp/push4_${PROFILE}_${n}_${arm}.out"
  local wd="/tmp/push4_${PROFILE}_${n}_${arm}.wd"
  local -a envs=("${COMMON_ENVS[@]}")
  [ "$arm" = "on" ] && envs+=(FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH4=1)

  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "    [dryrun] arm=$arm log=$log envs=${envs[*]}"
    echo "    [dryrun] cmd: python3 -m ftllm.cli benchmark ${COMMON[*]}"
    return 0
  fi

  timeout 500 tools/gpu_watchdog.sh "$log" --timeout 420 --label "push4_${n}_${arm}" -- \
      env "${envs[@]}" python3 -m ftllm.cli benchmark "${COMMON[@]}" > "$wd" 2>&1
  local rc=$?

  local total tpop ttft sha push4 launched outtok verdict ok="OK" why=""
  total=$(grep -a "Total time" "$log" 2>/dev/null | tail -1 | awk '{print $3}')
  tpop=$(grep -a "TPOP avg" "$log" 2>/dev/null | tail -1 | awk '{print $3}')
  ttft=$(grep -a "TTFT avg" "$log" 2>/dev/null | tail -1 | awk '{print $3}')
  sha=$(grep -a "Token stream sha256" "$log" 2>/dev/null | tail -1 | awk '{print $4}')
  outtok=$(grep -a "Actual output tokens" "$log" 2>/dev/null | tail -1 | awk '{print $4}')
  # 计数行是 "    <标签左对齐 32 字符><数字> call(s), ..."。两条标签词数不同，
  # 按空格取字段会错位（曾把 "kernel" 当成次数、把一次好跑判成作废），所以按标签切。
  push4=$(grep -a "launched on the TP4 push kernel" "$log" 2>/dev/null | tail -1 | sed 's/.*push kernel *//; s/ call.*//')
  launched=$(grep -a "launched on the custom kernel" "$log" 2>/dev/null | tail -1 | sed 's/.*custom kernel *//; s/ call.*//')
  verdict=$(grep -a GPU_WATCHDOG "$wd" 2>/dev/null | head -1 | sed 's/.*verdict=/verdict=/;s/ elapsed.*//')

  [ -z "$total" ] && { ok="VOID"; why="C1未过(无Total time)"; }
  # REF_SHA=auto 时，用第一次完成的 off 跑的哈希当参照（换 PROFILE 时用；比已知常量弱，
  # 只保证几次跑互相一致，仍能抓住数值漂移）。
  if [ "$REF_SHA" = "auto" ] && [ "$arm" = "off" ] && [ -n "$sha" ]; then
    REF_SHA="${sha:0:8}"
    echo "  （参照哈希取本次 off 跑：$REF_SHA）"
  fi
  if [ -n "$sha" ] && [ "${sha:0:8}" != "$REF_SHA" ]; then ok="VOID"; why="$why C2未过(哈希=${sha:0:8})"; fi
  # C4：别把"没吐出东西"当一次好跑。32K 输入那次吐了 0 个 token，
  # Total time 和哈希都在，按前三条会判成 OK——那是个假通过。
  if [ -z "$tpop" ] || [ "$tpop" = "n/a" ]; then ok="VOID"; why="$why C4未过(无TPOP，实际输出=${outtok:-无} token)"; fi
  if [ "$arm" = "on" ]; then
    if [ -z "$push4" ] || [ "$push4" = "0" ]; then ok="VOID"; why="$why C3未过(push4=${push4:-无})"; fi
  else
    if [ -n "$push4" ] && [ "$push4" != "0" ]; then ok="VOID"; why="$why 反向C3未过(off却跑了push4=$push4)"; fi
  fi
  local c
  c=$(cards)
  if [ "$c" != "4" ]; then ok="VOID"; why="$why 掉卡(卡数=$c)"; fi

  echo "  run$n-$arm rc=$rc Total=${total:-无} TPOP=${tpop:-无} TTFT=${ttft:-无} out=${outtok:-无} sha=${sha:0:8} push4=${push4:-无} custom=${launched:-无} $verdict 自验=$ok $why"
  [ "$ok" = "VOID" ] && return 90
  return 0
}

echo "=== push4 对照：顺序 [$SEQ] 参照哈希 $REF_SHA 开工前卡数=$(cards) Xid=$(xids) ==="
if [ "${DRYRUN:-0}" != "1" ]; then
  preflight || { echo "!! 开局自检未过，停手"; exit 2; }
fi

run=0
for arm in $SEQ; do
  run=$((run + 1))
  run_arm "$arm" "$run"
  if [ "$?" = "90" ]; then
    echo ">>> 第 $run 次作废（或掉卡），停手"
    break
  fi
done

echo "=== 汇总 (profile=$PROFILE) ==="
for arm in off on; do
  list=""
  for f in /tmp/push4_${PROFILE}_*_${arm}.out; do
    [ -f "$f" ] || continue
    v=$(grep -a "TPOP avg" "$f" 2>/dev/null | tail -1 | awk '{print $3}')
    [ -n "$v" ] && list="$list $v"
  done
  [ -z "$list" ] && { echo "$arm: 无有效样本"; continue; }
  med=$(echo $list | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {if (NR > 0) print a[int((NR+1)/2)]}')
  [ -z "$med" ] && { echo "$arm: 样本无法解析 [$list ]"; continue; }
  echo "$arm: TPOP 样本 [$list ] 中位 $med ms/token   对应 $(awk -v m="$med" 'BEGIN{printf "%.2f", 1000/m}') tok/s"
done
echo "=== 每次跑的计数行（C3 证据）==="
for f in /tmp/push4_${PROFILE}_*_off.out /tmp/push4_${PROFILE}_*_on.out; do
  [ -f "$f" ] || continue
  echo "$f"
  grep -a "launched on the custom kernel\|launched on the TP4 push kernel" "$f" 2>/dev/null | sed 's/^/   /'
done
echo "PUSH4ABDONE"
