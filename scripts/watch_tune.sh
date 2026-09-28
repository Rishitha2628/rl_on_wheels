#!/usr/bin/env bash
# Host-side watcher for the final chain: AIRL evaluations, the three-arm
# figure, and the TD3 tuning diagnostic.
#
# Each previous watcher exited once its own milestones had fired -- correct in
# isolation, but it means the durable notification channel keeps disappearing
# just as the next stage begins. This one watches the whole remainder of the
# work and only exits when the tuning diagnostic is done, which is the last
# thing scheduled.
#
#   systemctl --user enable --now rl-tune-watch
#   journalctl --user -u rl-tune-watch -f

set -uo pipefail

CONTAINER=sim
CHAIN_LOG=/logs/sweep/finish_and_tune.log
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
STATE="$STATE_DIR/tune_watch.state"
LOG="$STATE_DIR/tune_watch.log"
POLL=120

mkdir -p "$STATE_DIR"; touch "$STATE"

export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

log ()    { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
notify () { notify-send --urgency="$1" --app-name="rl_on_wheels" "$2" "$3" 2>/dev/null || true
            log "[$1] $2 — $3"; }
fired ()  { grep -qxF "$1" "$STATE"; }
mark ()   { echo "$1" >> "$STATE"; }
dexec ()  { docker exec "$CONTAINER" "$@" 2>/dev/null; }

log "tune watcher started (pid $$)"

while true; do
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    if ! fired container_gone; then
      mark container_gone
      notify critical "Sweep container stopped" "Results already written remain valid."
    fi
    sleep "$POLL"; continue
  fi

  if ! fired figure && dexec grep -q "regenerating the three-arm figure" "$CHAIN_LOG"; then
    mark figure
    notify normal "Three-arm figure regenerating" "BC, TD3 and AIRL — /logs/figures/"
  fi

  for c in a b; do
    if ! fired "tune_$c" && dexec test -f "/checkpoints/tune/td3_tune_${c}/eval_results.json"; then
      mark "tune_$c"
      best=$(dexec python3 -c "
import json
r = json.load(open('/checkpoints/tune/td3_tune_${c}/eval_results.json'))['results']
b = max(r, key=lambda x: x['success_rate'])
print('best %.1f%% at %s steps' % (b['success_rate']*100, f\"{b['budget_bridge_steps']:,}\"))
" 2>/dev/null)
      notify normal "Tuning run ${c} evaluated" "${best:-done} (baseline was 64.5%)"
    fi
  done

  if ! fired tune_done && dexec grep -q "TUNING DIAGNOSTIC COMPLETE" "$CHAIN_LOG"; then
    mark tune_done
    notify critical "TUNING DIAGNOSTIC DONE" \
      "Verdict table in /logs/sweep/finish_and_tune.log — does the plateau survive proper hyperparameters?"
    log "all milestones reported; exiting"
    exit 0
  fi

  # Silence must not read as success.
  if ! fired tune_done && ! fired chain_died \
     && ! dexec pgrep -f "finish_and_tune.sh" >/dev/null; then
    mark chain_died
    notify critical "Final chain stopped early" \
      "finish_and_tune.sh is gone with no completion marker. Check $CHAIN_LOG."
  fi

  sleep "$POLL"
done
