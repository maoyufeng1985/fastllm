#!/usr/bin/env bash
#
# safe_reboot.sh -- clear GPU benchmark stragglers out of the way before a reboot.
#
# Report mode is the default and is read-only. --clean runs the kill stages and
# stops. --reboot additionally reboots, using the sysrq reset when processes are
# unkillable.

set -uo pipefail

readonly SCRIPT_NAME='safe_reboot.sh'
readonly WATCHDOG_DROPIN='/etc/systemd/system.conf.d/99-safe-reboot.conf'
readonly SYSCTL_SYSRQ='/proc/sys/kernel/sysrq'
readonly SYSRQ_TRIGGER='/proc/sysrq-trigger'
readonly SYSRQ_REBOOT_BIT=128
readonly FIXED_PATTERNS=('ftllm\.cli' 'nsys profile' 'gpu_watchdog\.sh' 'nccl_bench' 'fi_vs_nccl')

MODE='report'
HARD=0
GRACE=20
KILL_WAIT=10
LOGFILE=''
WATCHDOG_SEC=''
EXTRA_PATTERNS=()
ASSUME_DEAD_CARD=0
NO_DEFAULT_PATTERNS=0
KILL_STAGES_SKIPPED=0

declare -A EXCLUDE=()
declare -A GPU_SET=()
declare -A PENDING=()
declare -a TARGET_PIDS=()
declare -a SURVIVORS=()
declare -a D_PIDS=()
declare -A TARGET_STATE=()
declare -A TARGET_CMD=()
declare -A TARGET_WHY=()

GPU_QUERY_NOTE=''
KERNEL_LOG_NOTE=''
DEAD_CARD_REASON=''
XID_LINES=''

usage() {
  cat <<'EOF'
tools/safe_reboot.sh [options]

  (no action flag)     REPORT ONLY. Change nothing, send no signals. This is the default.
  --clean              run the kill stages (SIGTERM, then SIGKILL), then stop. No reboot.
  --reboot             --clean, then actually reboot. If processes remain unkillable, use the
                       emergency path described below.
  --grace N            seconds to wait after SIGTERM before escalating (default 20)
  --kill-wait N        seconds to wait after SIGKILL (default 10)
  --pattern P          extra extended-regex matched against each process's cmdline; repeatable
  --no-default-patterns
                       drop the built-in name patterns; keep GPU holders and --pattern only
  --watchdog-sec N     before rebooting, set RebootWatchdogSec=N via a systemd drop-in
  --log FILE           also append output to FILE
  -h, --help           usage

Debug and test hooks, not part of normal operation:

  --hard               skip the SIGTERM stage and go straight to SIGKILL
  --assume-dead-card   force the dead-GPU-card verdict without probing the hardware,
                       so the skip-kill branch can be exercised on a healthy box

  SAFE_REBOOT_DEAD_CARD=no|yes overrides the dead-card verdict. Only for testing the
  kill path on a box whose kernel log still holds old Xid 79/154 lines.

Exit codes: 0 clean, 1 unkillable (wedge) processes remain, 2 usage error, 3 not root.
EOF
}

log() {
  local line
  line="[$(date +%H:%M:%S)] $*"
  if [[ -n "$LOGFILE" ]]; then
    printf '%s\n' "$line" | tee -a "$LOGFILE"
  else
    printf '%s\n' "$line"
  fi
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
  usage >&2
  exit 2
}

need_int() {
  [[ ${2:-} =~ ^[0-9]+$ ]] || usage_error "$1 requires a non-negative integer"
}

parse_args() {
  while (( $# )); do
    case "$1" in
      --clean)            MODE='clean'; shift ;;
      --reboot)           MODE='reboot'; shift ;;
      --hard)             HARD=1; shift ;;
      --grace)            need_int "$1" "${2:-}"; GRACE=$2; shift 2 ;;
      --kill-wait)        need_int "$1" "${2:-}"; KILL_WAIT=$2; shift 2 ;;
      --watchdog-sec)     need_int "$1" "${2:-}"; WATCHDOG_SEC=$2; shift 2 ;;
      --pattern)
        [[ -n "${2:-}" ]] || usage_error '--pattern requires a non-empty extended regex'
        EXTRA_PATTERNS+=("$2"); shift 2 ;;
      --no-default-patterns) NO_DEFAULT_PATTERNS=1; shift ;;
      --log)
        [[ -n "${2:-}" ]] || usage_error '--log requires a file path'
        LOGFILE=$2; shift 2 ;;
      --assume-dead-card) ASSUME_DEAD_CARD=1; shift ;;
      -h|--help)          usage; exit 0 ;;
      --)                 shift; break ;;
      *)                  usage_error "unknown option: $1" ;;
    esac
  done
  (( $# == 0 )) || usage_error "unexpected argument: $1"
}

init_log() {
  [[ -n "$LOGFILE" ]] || return 0
  if ! : >> "$LOGFILE" 2>/dev/null; then
    printf '%s: cannot append to log file: %s\n' "$SCRIPT_NAME" "$LOGFILE" >&2
    exit 2
  fi
}

# --- process identity ------------------------------------------------------

state_of() {
  awk '/^State:/{print $2}' "/proc/$1/status" 2>/dev/null
}

# A zombie is already dead: it only awaits its parent's wait(). Counting it as a
# survivor would report a successful kill as a wedge.
is_live() {
  [[ -d /proc/$1 ]] || return 1
  [[ "$(state_of "$1")" != 'Z' ]]
}

# Field 4 of /proc/<pid>/stat is not the ppid. Field 2 is the comm inside
# parentheses and may contain spaces and parens, which shifts every later field.
ppid_of() {
  awk '/^PPid:/{print $2}' "/proc/$1/status" 2>/dev/null
}

# The script must never target its own shell or anything above it: a bad ancestor
# walk here means SIGKILLing the session that launched us.
collect_exclusions() {
  local pid=$$ parent
  EXCLUDE[$$]=1
  EXCLUDE[1]=1
  while (( pid > 1 )); do
    [[ -r /proc/$pid/status ]] || break
    parent="$(ppid_of "$pid")"
    [[ "$parent" =~ ^[0-9]+$ ]] || break
    (( parent > 0 )) || break
    EXCLUDE[$parent]=1
    pid=$parent
  done
}

# --- target selection ------------------------------------------------------

collect_gpu_pids() {
  local out rc line
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)"; rc=$?
  if (( rc != 0 )); then
    GPU_QUERY_NOTE="nvidia-smi --query-compute-apps failed (rc=$rc): $(printf '%s' "$out" | tr '\n' ';' | sed 's/  */ /g')"
    return 0
  fi
  while IFS= read -r line; do
    line="${line//[[:space:]]/}"
    [[ "$line" =~ ^[0-9]+$ ]] || continue
    GPU_SET[$line]=1
  done <<< "$out"
  (( ${#GPU_SET[@]} )) || GPU_QUERY_NOTE='nvidia-smi reported no compute applications'
}

collect_targets() {
  local d pid cmdline pattern why
  local -a builtin_reasons
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    [[ -r "$d/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)" || continue
    cmdline="${cmdline% }"
    [[ -n "$cmdline" ]] || continue

    [[ -z "${EXCLUDE[$pid]:-}" ]] || continue
    [[ "$cmdline" != *"$SCRIPT_NAME"* ]] || continue

    why=''
    [[ -n "${GPU_SET[$pid]:-}" ]] && why='gpu-holder'
    if (( ! NO_DEFAULT_PATTERNS )); then
      builtin_reasons=()
      for pattern in "${FIXED_PATTERNS[@]}"; do
        if printf '%s' "$cmdline" | grep -Eq -- "$pattern"; then
          builtin_reasons+=("$pattern")
        fi
      done
      (( ${#builtin_reasons[@]} )) && why="${why:+$why,}builtin-pattern($(IFS='|'; echo "${builtin_reasons[*]}"))"
    fi
    for pattern in "${EXTRA_PATTERNS[@]}"; do
      if printf '%s' "$cmdline" | grep -Eq -- "$pattern"; then
        why="${why:+$why,}extra-pattern($pattern)"
      fi
    done
    [[ -n "$why" ]] || continue

    TARGET_PIDS+=("$pid")
    TARGET_CMD[$pid]="$cmdline"
    TARGET_STATE[$pid]="$(state_of "$pid")"
    TARGET_WHY[$pid]="$why"
  done
}

print_pattern_sources() {
  local joined=''
  log '=== active pattern sources ==='
  if (( NO_DEFAULT_PATTERNS )); then
    log '  builtin-pattern: DISABLED by --no-default-patterns'
  else
    log "  builtin-pattern: ${FIXED_PATTERNS[*]}"
  fi
  if (( ${#EXTRA_PATTERNS[@]} )); then
    joined="$(printf '%s|' "${EXTRA_PATTERNS[@]}")"
    log "  extra-pattern: ${joined%|}"
  else
    log '  extra-pattern: none'
  fi
  log '  gpu-holder: always active (nvidia-smi --query-compute-apps)'
}

# --- hardware fault check --------------------------------------------------

# Killing a task that holds a GPU the driver has already lost is the documented
# way to turn a single-card fault into a whole-line outage, because the kernel
# then waits forever on the unresponsive device.
# Real lines read "NVRM: Xid (PCI:0000:08:00): 79, ...", so digits from the PCI
# address sit between "Xid" and the code. Match the code on a non-digit boundary
# instead of on whatever text follows "Xid".
kernel_xid_lines() {
  local out
  out="$(dmesg 2>/dev/null)" || out="$(journalctl -k -b --no-pager 2>/dev/null)" || true
  if [[ -z "$out" ]]; then
    KERNEL_LOG_NOTE='kernel log unreadable (both dmesg and journalctl -k failed)'
    return 0
  fi
  printf '%s\n' "$out" | grep -aE 'Xid.*[^0-9](79|154)([^0-9]|$)' || true
}

detect_card_state() {
  local choice="${SAFE_REBOOT_DEAD_CARD:-auto}" enum rc xids

  if (( ASSUME_DEAD_CARD )); then
    DEAD_CARD_REASON='forced by --assume-dead-card'
    return 0
  fi
  case "$choice" in
    auto) ;;
    yes|force|always) DEAD_CARD_REASON="forced by SAFE_REBOOT_DEAD_CARD=$choice"; return 0 ;;
    no|never|off)
      DEAD_CARD_REASON=''
      log "dead-card check disabled by SAFE_REBOOT_DEAD_CARD=$choice (test hook)"
      return 0 ;;
    *) log "WARNING: SAFE_REBOOT_DEAD_CARD='$choice' not understood; using auto" ;;
  esac

  enum="$(nvidia-smi -L 2>&1)"; rc=$?
  if (( rc != 0 )) || ! grep -q '^GPU ' <<< "$enum"; then
    DEAD_CARD_REASON="nvidia-smi cannot enumerate any GPU (rc=$rc)"
  fi

  xids="$(kernel_xid_lines)"
  if [[ -n "$xids" ]]; then
    XID_LINES="$xids"
    DEAD_CARD_REASON="${DEAD_CARD_REASON:+$DEAD_CARD_REASON; }Xid 79/154 in the kernel log since boot"
  fi
}

# --- reporting -------------------------------------------------------------

print_preflight() {
  local wd found=0
  log '=== preflight ==='
  log "uptime: $(uptime -p 2>/dev/null || uptime)"
  log "kernel: $(uname -r)"
  for wd in /dev/watchdog*; do
    [[ -e "$wd" ]] || continue
    log "watchdog device: $wd"
    found=1
  done
  (( found )) || log 'watchdog device: none of /dev/watchdog* present'
  log "RebootWatchdogUSec: $(systemctl show -p RebootWatchdogUSec 2>/dev/null || echo unknown)"
}

print_card_state() {
  log '=== dead-GPU-card check ==='
  if [[ -n "$DEAD_CARD_REASON" ]]; then
    log "DEAD CARD: $DEAD_CARD_REASON"
  else
    log 'GPUs enumerate and no Xid 79/154 in the kernel log.'
  fi
  [[ -n "$GPU_QUERY_NOTE" ]] && log "nvidia-smi: $GPU_QUERY_NOTE"
  [[ -n "$KERNEL_LOG_NOTE" ]] && log "kernel log: $KERNEL_LOG_NOTE"
  if [[ -n "$XID_LINES" ]]; then
    log 'matching kernel log lines:'
    printf '%s\n' "$XID_LINES" | head -n 20 | while IFS= read -r line; do
      log "  $line"
    done
  fi
}

print_targets() {
  local pid gpu
  log '=== targets ==='
  if (( ${#TARGET_PIDS[@]} == 0 )); then
    log '  none'
    return 0
  fi
  log "$(printf '%-8s %-6s %-4s %-34s %s' 'PID' 'STATE' 'GPU' 'REASON' 'CMDLINE')"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    gpu='-'; [[ -n "${GPU_SET[$pid]:-}" ]] && gpu='yes'
    log "$(printf '%-8s %-6s %-4s %-34s %s' "$pid" "${TARGET_STATE[$pid]:-?}" "$gpu" "$(printf '%.34s' "${TARGET_WHY[$pid]:-}")" "$(printf '%.110s' "${TARGET_CMD[$pid]:-}")")"
  done < <(printf '%s\n' "${TARGET_PIDS[@]}" | sort -n)
}

scan_d_state() {
  local d pid
  D_PIDS=()
  for d in /proc/[0-9]*; do
    pid="${d#/proc/}"
    [[ "$(state_of "$pid")" == 'D' ]] || continue
    D_PIDS+=("$pid")
  done
}

print_d_state() {
  local pid comm cmdline
  log '=== processes in D state (system-wide) ==='
  if (( ${#D_PIDS[@]} == 0 )); then
    log '  none'
    return 0
  fi
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    comm="$(cat "/proc/$pid/comm" 2>/dev/null)"
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    cmdline="${cmdline% }"
    log "  pid $pid [$comm] ${cmdline:-kthread}"
  done < <(printf '%s\n' "${D_PIDS[@]}" | sort -n)
  log '  D state ignores SIGKILL; these are what make systemd-shutdown wait.'
}

# --- kill stages -----------------------------------------------------------

poll_exits() {
  local label=$1 ticks=$2 i pid
  for (( i=0; i<ticks; i++ )); do
    (( ${#PENDING[@]} )) || return 0
    sleep 0.5
    for pid in "${!PENDING[@]}"; do
      is_live "$pid" && continue
      log "  [$label] pid $pid exited"
      unset 'PENDING[$pid]'
    done
  done
}

run_kill_stages() {
  local pid

  if [[ -n "$DEAD_CARD_REASON" ]]; then
    KILL_STAGES_SKIPPED=1
    log "SKIPPING the SIGTERM and SIGKILL stages: $DEAD_CARD_REASON"
    log 'Rule: never SIGKILL a process holding a GPU the driver has lost (Xid 79/154).'
    log 'The kill cannot free the card and is the documented way to escalate a single-card fault.'
    return 0
  fi

  PENDING=()
  for pid in "${TARGET_PIDS[@]}"; do PENDING[$pid]=1; done
  if (( ${#PENDING[@]} == 0 )); then
    log 'no targets to signal'
    return 0
  fi

  if (( ! HARD )); then
    log "stage 1: SIGTERM to ${#PENDING[@]} target(s), then up to ${GRACE}s grace"
    for pid in "${!PENDING[@]}"; do
      if kill -TERM "$pid" 2>/dev/null; then
        log "  SIGTERM -> pid $pid"
      else
        log "  SIGTERM -> pid $pid failed (already gone or gone at once)"
      fi
    done
    poll_exits 'grace' "$(( GRACE * 2 ))"
  else
    log 'stage 1 skipped (--hard)'
  fi

  if (( ${#PENDING[@]} )); then
    log "stage 2: SIGKILL to ${#PENDING[@]} survivor(s), then up to ${KILL_WAIT}s"
    for pid in "${!PENDING[@]}"; do
      if kill -KILL "$pid" 2>/dev/null; then
        log "  SIGKILL -> pid $pid"
      else
        log "  SIGKILL -> pid $pid failed"
      fi
    done
    poll_exits 'kill' "$(( KILL_WAIT * 2 ))"
  else
    log 'stage 2 skipped: nothing survived SIGTERM'
  fi
}

collect_survivors() {
  local pid
  SURVIVORS=()
  for pid in "${TARGET_PIDS[@]}"; do
    is_live "$pid" && SURVIVORS+=("$pid")
  done
}

print_wedge_report() {
  local pid cmd
  log '=== wedge report ==='
  if (( KILL_STAGES_SKIPPED )); then
    log '  the kill stages were skipped, so nothing was signalled; the targets below are'
    log '  the ones that would have been signalled:'
  fi
  if (( ${#SURVIVORS[@]} == 0 )); then
    log '  surviving targets: none'
  else
    log "  surviving targets (${#SURVIVORS[@]}):"
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
      log "    pid $pid [$(state_of "$pid")] $(printf '%.100s' "${cmd% }")"
    done < <(printf '%s\n' "${SURVIVORS[@]}" | sort -n)
  fi
  print_d_state
}

# --- reboot decision -------------------------------------------------------

apply_watchdog_drop_in() {
  local dir
  dir="$(dirname -- "$WATCHDOG_DROPIN")"
  if ! mkdir -p -- "$dir"; then
    log "ERROR: cannot create $dir"
    return 1
  fi
  if ! printf '[Manager]\nRebootWatchdogSec=%s\n' "$WATCHDOG_SEC" > "$WATCHDOG_DROPIN"; then
    log "ERROR: cannot write $WATCHDOG_DROPIN"
    return 1
  fi
  log "wrote $WATCHDOG_DROPIN with RebootWatchdogSec=$WATCHDOG_SEC"
  if ! systemctl daemon-reexec; then
    log 'WARNING: systemctl daemon-reexec failed; the drop-in applies at the next daemon load'
  fi
  log 'This applies to subsequent shutdowns, not to one already in flight.'
}

emergency_sysrq_reboot() {
  local old new rc=0

  log 'emergency path: sync, then sysrq reset'
  if ! sync; then
    log 'WARNING: sync returned non-zero; resetting anyway'
  fi
  log 'sync complete'

  old="$(tr -d '[:space:]' < "$SYSCTL_SYSRQ" 2>/dev/null)"
  if ! [[ "$old" =~ ^[0-9]+$ ]]; then
    log "ERROR: cannot read a numeric mask from $SYSCTL_SYSRQ"
    return 1
  fi
  new=$(( old | SYSRQ_REBOOT_BIT ))
  log "sysrq mask: previous=$old new=$new"

  if (( (old & SYSRQ_REBOOT_BIT) == 0 )); then
    if ! printf '%s\n' "$new" > "$SYSCTL_SYSRQ"; then
      log "ERROR: cannot write $new to $SYSCTL_SYSRQ"
      return 1
    fi
    log "sysrq reboot bit (128) enabled, mask $old -> $new"
  else
    log 'sysrq reboot bit already set'
  fi

  log 'writing b to the sysrq trigger: immediate reset, no further cleanup'
  if ! printf 'b\n' > "$SYSRQ_TRIGGER"; then
    log "ERROR: write to $SYSRQ_TRIGGER failed; restoring sysrq mask to $old"
    if printf '%s\n' "$old" > "$SYSCTL_SYSRQ"; then
      log "sysrq mask restored to $old"
    else
      log "WARNING: could not restore sysrq mask $old"
    fi
    rc=1
  fi
  return $rc
}

decide_reboot() {
  local wd

  if (( ${#SURVIVORS[@]} == 0 && ${#D_PIDS[@]} == 0 )); then
    log 'No surviving targets and no D-state processes.'
    if [[ -n "$WATCHDOG_SEC" && "$MODE" == 'reboot' ]]; then
      apply_watchdog_drop_in || true
    fi
    if [[ "$MODE" != 'reboot' ]]; then
      log '--clean requested no reboot; stopping here.'
      exit 0
    fi
    sync || log 'WARNING: sync returned non-zero'
    log 'sync complete; rebooting with systemctl reboot'
    if ! systemctl reboot; then
      log 'ERROR: systemctl reboot failed'
      exit 1
    fi
    exit 0
  fi

  log 'WEDGE: processes will not be reaped.'
  wd="$(systemctl show -p RebootWatchdogUSec --value 2>/dev/null)"
  [[ -n "$wd" ]] || wd='unknown'
  log "systemd-shutdown will SIGKILL the survivors, fail to reap them (D state ignores SIGKILL), and wait for the hardware watchdog to reset the box after RebootWatchdogUSec=$wd."
  log "${#SURVIVORS[@]} surviving target(s), ${#D_PIDS[@]} D-state process(es) system-wide."

  if [[ -n "$WATCHDOG_SEC" ]]; then
    apply_watchdog_drop_in || true
  fi

  if [[ "$MODE" == 'reboot' ]]; then
    emergency_sysrq_reboot || exit 1
    exit 0
  fi

  log "Next command for the operator: $0 --reboot${WATCHDOG_SEC:+ --watchdog-sec $WATCHDOG_SEC}"
  exit 1
}

main() {
  parse_args "$@"

  if (( EUID != 0 )); then
    printf '%s: must run as root (exit 3)\n' "$SCRIPT_NAME" >&2
    exit 3
  fi

  init_log
  collect_exclusions
  collect_gpu_pids
  detect_card_state
  collect_targets

  log "safe_reboot.sh mode=$MODE grace=$GRACE kill_wait=$KILL_WAIT hard=$HARD"
  print_preflight
  print_pattern_sources
  print_targets
  print_card_state
  scan_d_state
  print_d_state

  if [[ "$MODE" == 'report' ]]; then
    log '=== report only: no signals sent, no state changed ==='
    exit 0
  fi

  run_kill_stages
  collect_survivors
  scan_d_state
  print_wedge_report
  decide_reboot
}

main "$@"
