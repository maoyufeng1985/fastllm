#!/bin/bash
# GPU 看门狗：包住一条多卡测试命令，同时监控 GPU 异常，判定挂住就抓现场并杀掉。
#
# 用法:
#   tools/gpu_watchdog.sh <日志文件> [--timeout 秒] [--label 名字] -- <命令...>
#
# 例:
#   tools/gpu_watchdog.sh /tmp/r.out --timeout 600 --label r1 -- \
#       env FOO=1 python3 -m ftllm.cli benchmark ...
#
# 退出码:
#   0    命令正常结束（不看结果对不对，只看有没有挂）
#   90   判定卡死，已抓现场并杀掉
#   124  超过 --timeout
#   其他 命令自身的退出码
#
# 判定规则（为什么是这条）：TP 集合通信挂住时，卡住的 rank 没有任何 kernel（util=0），
# 其余 rank 在 NCCL 里空转（util=100%）。这跟"模型加载中全 0%""跑完全 0%"都不同。
#
# 现场写到 <日志文件去后缀>.gpuwatch.txt：逐卡 util/显存时间线 + 目标进程每个线程的
# wchan/syscall（区分卡在 CUDA 调用还是主机自旋）+ 日志尾部。还会比对跑前跑后的 Xid 计数。
set -u

LOG="${1:?用法: gpu_watchdog.sh <日志文件> [--timeout 秒] [--label 名字] -- <命令...>}"
shift

TIMEOUT="${GPU_WATCHDOG_TIMEOUT:-900}"
SAMPLE="${GPU_WATCHDOG_SAMPLE:-5}"
STRIKES="${GPU_WATCHDOG_STRIKES:-4}"
LABEL="run"
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --label)   LABEL="$2";   shift 2 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done
[ $# -gt 0 ] || { echo "gpu_watchdog: 没有给命令" >&2; exit 2; }

DIAG="${LOG%.out}.gpuwatch.txt"
: > "$DIAG"

xid_count() { dmesg 2>/dev/null | grep -ci "xid" || true; }
gpu_triplet() { nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd';'; }

XID_BEFORE=$(xid_count)
PERM_BEFORE=$(nvidia-smi -q 2>/dev/null | grep -E "Pending Page Blacklist|Double Bit ECC" | tr -d ' ' | paste -sd';')

setsid "$@" >"$LOG" 2>&1 &
CMD_PID=$!

deepest_python_pid() {
    local pid=$CMD_PID depth
    for depth in 1 2 3; do
        local child
        child=$(pgrep -P "$pid" 2>/dev/null | head -1)
        [ -n "$child" ] || break
        pid=$child
    done
    echo "$pid"
}

capture_scene() {
    local why="$1" pid
    pid=$(deepest_python_pid)
    {
        echo "=========== GPU WATCHDOG 现场 ($LABEL) ==========="
        echo "时间: $(date '+%F %T')"
        echo "判定: $why"
        echo
        echo "--- 1. 逐卡现状 (index,util%,memMiB) ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
                   --format=csv,noheader
        echo
        echo "--- 2. 逐卡时间线（最近 ${STRIKES} 次采样） ---"
        tail -n "$STRIKES" "$DIAG.timeline" 2>/dev/null
        echo
        echo "--- 3. 目标进程线程阻塞点 (pid=$pid) ---"
        if [ -d "/proc/$pid" ]; then
            for t in /proc/"$pid"/task/*; do
                tid=$(basename "$t")
                printf 'tid=%-8s state=%-2s wchan=%-34s syscall=%-4s\n' \
                    "$tid" \
                    "$(awk '/^State/{print $2}' "$t/status" 2>/dev/null)" \
                    "$(cat "$t/wchan" 2>/dev/null)" \
                    "$(awk '{print $1}' "$t/syscall" 2>/dev/null)"
            done
            echo "  线程状态统计: $(for t in /proc/$pid/task/*; do awk '/^State/{print $2}' $t/status 2>/dev/null; done | sort | uniq -c | tr '\n' ' ')"
        else
            echo "  (进程 $pid 已不存在)"
        fi
        echo
        echo "--- 4. 日志尾部 (最后 30 行) ---"
        tail -n 30 "$LOG" 2>/dev/null
        echo
        echo "--- 5. Xid ---"
        echo "跑前计数=$XID_BEFORE  现在计数=$(xid_count)"
        [ "$(xid_count)" != "$XID_BEFORE" ] && dmesg 2>/dev/null | grep -i xid | tail -5
        echo "================================================="
    } >> "$DIAG" 2>&1
}

kill_tree() {
    local pgid
    pgid=$(ps -o pgid= -p "$CMD_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ]; then kill -9 -"$pgid" 2>/dev/null; fi
    kill -9 "$CMD_PID" 2>/dev/null
}

verdict="" ; strike=0 ; elapsed=0 ; poll="$SAMPLE"
while kill -0 "$CMD_PID" 2>/dev/null; do
    # 每 1 秒看一次进程，每 SAMPLE 秒采一次 GPU（保持轮询粒度细，避免长 sleep）
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $((elapsed % poll)) -eq 0 ]; then
        row=$(gpu_triplet)
        ts=$(date '+%T')
        line="$ts  $row"
        echo "$line" >> "$DIAG.timeline"
        busy=$(echo "$row" | tr ';' '\n' | awk -F, '$2>=50' | wc -l)
        idle=$(echo "$row" | tr ';' '\n' | awk -F, '$2==0' | wc -l)
        if [ "$busy" -ge 2 ] && [ "$idle" -ge 1 ]; then
            strike=$((strike + 1))
        else
            strike=0
        fi
        if [ "$strike" -ge "$STRIKES" ]; then
            verdict="HANG"
            capture_scene "疑似卡死: ${strike} 次连续采样出现「≥2 卡在跑、≥1 卡 util=0」($row)"
            kill_tree
            break
        fi
        # 掉卡：nvidia-smi 连设备句柄都拿不到（Xid 79/154 的特征）。
        # 这个判据必须放在轮内：掉卡时测试进程常常还卡在 CUDA 调用里不退出，
        # 若等"轮末再判"，看门狗会空转到满超时，白等几分钟。
        case "$row" in
            *"Unabletodeterminethedevicehandle"*|*"Unable to determine the device handle"*|*"No devices were found"*)
                verdict="DEAD_GPU"
                capture_scene "掉卡: nvidia-smi 拿不到设备句柄（Xid 79/154 特征）row=$row"
                kill_tree
                break
                ;;
        esac
    fi
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        verdict="TIMEOUT"
        capture_scene "超过 --timeout ${TIMEOUT}s"
        kill_tree
        break
    fi
done

if [ -z "$verdict" ]; then
    wait "$CMD_PID" 2>/dev/null
    rc=$?
else
    wait "$CMD_PID" 2>/dev/null
    rc=90
    [ "$verdict" = "TIMEOUT" ] && rc=124
fi

# 收工是否干净：给 10 秒把显存放掉
leak=""
for _ in $(seq 1 10); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
    echo "$used" | grep -qE '[1-9]' || { leak=""; break; }
    leak="$used"
    sleep 1
done
XID_AFTER=$(xid_count)

{
    echo "--- 6. 收工 ---"
    echo "退出码=$rc  判定=${verdict:-OK}  用时约 ${elapsed}s"
    echo "Xid: 跑前=$XID_BEFORE 跑后=$XID_AFTER $([ "$XID_AFTER" != "$XID_BEFORE" ] && echo '<<< 有新增，异常！')"
    echo "ECC/黑名单: 跑前=$PERM_BEFORE"
    echo "           跑后=$(nvidia-smi -q 2>/dev/null | grep -E 'Pending Page Blacklist|Double Bit ECC' | tr -d ' ' | paste -sd';')"
    if [ -n "$leak" ]; then
        echo "显存未释放 <<< 异常！剩余: $leak MiB"
    else
        echo "显存已归零"
    fi
} >> "$DIAG" 2>&1

echo "GPU_WATCHDOG verdict=${verdict:-OK} exit=$rc elapsed=${elapsed}s log=$LOG diag=$DIAG"
exit "$rc"
