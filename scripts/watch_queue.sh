#!/usr/bin/env bash
# Watches the whole remaining queue: TD3 confirmation, the AIRL arm, the
# reward variants. Exits only when the queue reports complete.
#
# Scoped to everything rather than one stage. Earlier watchers each covered a
# single stage and exited when it finished, so the durable channel lapsed just
# as the next stage began -- four times. One of those lapses hid a chain that
# had died, because a *later* stage was still running and the "nothing is
# running" check stayed quiet. This one keys off the queue's own log markers,
# so a stage dying is visible even while another runs.
set -uo pipefail

CONTAINER=sim
QLOG=/logs/sweep/queue.log
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
STATE="$STATE_DIR/queue_watch.state"
LOG="$STATE_DIR/queue_watch.log"
POLL=180

mkdir -p "$STATE_DIR"; touch "$STATE"
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

log ()    { printf '%s  %s\n' "$(date '+%F %H:%M:%S')" "$*" >> "$LOG"; }
notify () { notify-send --urgency="$1" --app-name="rl_on_wheels" "$2" "$3" 2>/dev/null || true; log "[$1] $2 — $3"; }
fired ()  { grep -qxF "$1" "$STATE"; }
mark ()   { echo "$1" >> "$STATE"; }
dexec ()  { docker exec "$CONTAINER" "$@" 2>/dev/null; }

log "queue watcher started (pid $$)"

while true; do
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    if ! fired container_gone; then
      mark container_gone
      notify critical "Container stopped" "Queue halted (reboot?). Restart with: docker start sim && docker exec -d sim /scripts/run_queue.sh"
    fi
    sleep "$POLL"; continue
  fi

  for s in 0 1 2; do
    if ! fired "td3_$s" && dexec test -f "/checkpoints/tune/td3_confirm_seed${s}/eval_results.json"; then
      mark "td3_$s"
      notify normal "TD3 confirm seed $s" \
        "$(dexec grep -o "TD3 confirm seed $s: .*" "$QLOG" | tail -1 | sed "s/.*: //")  (tuning saw 79.0%, BC 63.7%)"
    fi
  done
  if ! fired td3_done && dexec grep -q "TD3 CONFIRMATION COMPLETE" "$QLOG"; then
    mark td3_done; notify critical "TD3 confirmation done" "Three seeds. Does 79% hold?"
  fi
  if ! fired airl_done && dexec grep -q "AIRL arm complete" "$QLOG"; then
    mark airl_done; notify critical "AIRL arm complete" "All three seeds on the fixed reward."
  fi
  for v in c d; do
    if ! fired "rv_$v" && dexec test -f "/checkpoints/tune/td3_reward_${v}/eval_results.json"; then
      mark "rv_$v"; notify normal "Reward variant $v done" "See $QLOG"
    fi
  done
  if ! fired queue_done && dexec grep -q "QUEUE COMPLETE" "$QLOG"; then
    mark queue_done; notify critical "ALL QUEUED WORK COMPLETE" "Figure regenerated. /logs/figures/"
    log "exiting"; exit 0
  fi
  # ── stall detection ─────────────────────────────────────────────────────────
  # Checking that a process EXISTS is not the same as checking it is doing
  # anything. An AIRL seed sat blocked for 39 hours with four simulators
  # burning six cores beside it, no errors logged, and this watcher silent --
  # because the process was still there. Progress is the thing to watch.
  cur=$(dexec bash -c '''for f in /logs/sweep/airl_seed*.log /logs/sweep/td3_confirm_seed*.log; do
          [ -f "$f" ] || continue
          printf "%s " "$(grep -E "^\|    total_timesteps" "$f" 2>/dev/null | tail -1 | awk "{print \$4}")"
        done''' 2>/dev/null)
  if [[ -n "${cur:-}" ]]; then
    if [[ "${cur}" == "${LAST_PROGRESS:-}" ]]; then
      STALL_TICKS=$(( ${STALL_TICKS:-0} + 1 ))
    else
      STALL_TICKS=0; LAST_PROGRESS="$cur"
    fi
    # POLL is 180s, so 12 ticks is ~36 minutes of no movement anywhere.
    if [[ ${STALL_TICKS:-0} -ge 12 ]] && ! fired stalled; then
      mark stalled
      notify critical "Training appears STALLED" \
        "No step-count movement in ~36 min while processes are still alive. Check: docker exec sim tail /logs/sweep/queue.log"
    fi
  fi

  if ! fired queue_died && ! dexec pgrep -f "run_queue.sh" >/dev/null; then
    mark queue_died
    notify critical "Queue stopped early" "run_queue.sh is gone with no completion marker. Check $QLOG."
    exit 0
  fi
  sleep "$POLL"
done
