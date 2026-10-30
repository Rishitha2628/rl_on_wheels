#!/usr/bin/env bash
# Disk and thermal watchdog for long runs.
#
# Separate from the progress watchers: those report milestones, this one reports
# the machine getting into trouble. Disk is the dangerous one -- the root
# filesystem has been sitting above 90% used, and a training run that fills it
# corrupts whatever it was writing at the time.
set -uo pipefail

DISK_WARN_GB=4          # notify below this
DISK_CRIT_GB=2          # notify loudly; run_substrate.sh stops itself here too
GPU_WARN_C=80
CPU_WARN_C=92
POLL=300

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
LOG="$STATE_DIR/health.log"
mkdir -p "$STATE_DIR"

export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

log ()    { printf '%s  %s\n' "$(date '+%F %H:%M:%S')" "$*" >> "$LOG"; }
notify () { notify-send --urgency="$1" --app-name="rl_on_wheels" "$2" "$3" 2>/dev/null || true; log "[$1] $2 -- $3"; }

last_disk_state=ok; last_gpu_state=ok; last_cpu_state=ok
log "health watcher started (pid $$)"

while true; do
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  gpu_c=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -dc '0-9')
  cpu_c=$(awk '{printf "%.0f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)

  # Disk. Report each transition once rather than every poll.
  if   (( free_gb < DISK_CRIT_GB )); then state=crit
  elif (( free_gb < DISK_WARN_GB )); then state=warn
  else state=ok; fi
  if [[ "$state" != "$last_disk_state" ]]; then
    case "$state" in
      crit) notify critical "Disk critically low" "${free_gb}G free on /. Runs will stop rather than fill it." ;;
      warn) notify normal   "Disk getting low"    "${free_gb}G free on /." ;;
      ok)   [[ "$last_disk_state" != ok ]] && notify low "Disk recovered" "${free_gb}G free." ;;
    esac
    last_disk_state=$state
  fi

  if [[ -n "${gpu_c:-}" ]] && (( gpu_c > GPU_WARN_C )); then
    [[ "$last_gpu_state" == ok ]] && notify normal "GPU hot" "${gpu_c}C"
    last_gpu_state=hot
  else
    last_gpu_state=ok
  fi

  if [[ -n "${cpu_c:-}" ]] && (( cpu_c > CPU_WARN_C )); then
    [[ "$last_cpu_state" == ok ]] && notify normal "CPU hot" "${cpu_c}C (throttling protects it, but it is running warm)"
    last_cpu_state=hot
  else
    last_cpu_state=ok
  fi

  log "disk=${free_gb}G gpu=${gpu_c:-?}C cpu=${cpu_c:-?}C"
  sleep "$POLL"
done
