#!/usr/bin/env bash
# Host-side watcher for the AIRL arm. Desktop notifications on each seed and
# on completion.
#
# Separate from watch_sweep.sh because that one exits once its milestones have
# all fired -- correct behaviour, but it means the durable notification channel
# disappears the moment the TD3 arm finishes. This restores it for the AIRL
# arm, which runs for roughly another 17 hours.
#
# Runs on the host rather than inside the container, so it survives the
# terminal being closed.
#
#   systemctl --user daemon-reload
#   systemctl --user enable --now rl-airl-watch
#   journalctl --user -u rl-airl-watch -f

set -uo pipefail

CONTAINER=sim
SWEEP_DIR=/checkpoints/sweep
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
STATE="$STATE_DIR/airl_watch.state"
LOG="$STATE_DIR/airl_watch.log"
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

log "airl watcher started (pid $$)"

while true; do
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    if ! fired container_gone; then
      mark container_gone
      notify critical "Sweep container stopped" \
        "The '$CONTAINER' container is not running. Checkpoints already written remain valid."
    fi
    sleep "$POLL"; continue
  fi

  for s in 0 1 2; do
    f="$SWEEP_DIR/airl_seed${s}/run_summary.json"
    if ! fired "airl_seed_$s" && dexec test -f "$f"; then
      mark "airl_seed_$s"
      info=$(dexec python3 -c "
import json
d = json.load(open('$f'))
print('%.1f h, %d budgets' % (d['wall_total_h'], len(d['budgets_reached'])))
" 2>/dev/null)
      notify normal "AIRL seed $s complete" "${info:-done}"
    fi
  done

  if ! fired airl_done && dexec grep -q "AIRL sweep finished" /logs/sweep/after_sweep.log; then
    mark airl_done
    notify critical "AIRL SWEEP FINISHED" \
      "All arms collected. Next: eval_budget.py on airl_seed*, then analyze_crossover.py for the three-arm figure."
    log "all milestones reported; exiting"
    exit 0
  fi

  # Silence must not read as progress: if the chain and the trainer are both
  # gone with no completion marker, say so.
  if ! fired airl_done && ! fired airl_died \
     && ! dexec pgrep -f "run_after_sweep.sh|run_sweep_airl.sh|sweep_airl.py" >/dev/null; then
    mark airl_died
    notify critical "AIRL sweep stopped early" \
      "No trainer or orchestrator running and no completion marker. Check /logs/sweep/after_sweep.log — budget checkpoints already written remain valid."
  fi

  sleep "$POLL"
done
