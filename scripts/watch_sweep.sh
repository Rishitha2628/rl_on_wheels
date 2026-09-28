#!/usr/bin/env bash
# Host-side watcher for the interaction-budget sweep. Sends desktop
# notifications as milestones land.
#
# This runs on the host rather than inside the container, so it survives the
# terminal being closed. The sweep takes roughly 44 hours, which is longer than
# any terminal session is likely to stay open.
#
# State is kept in a file so that a restart (or a systemd auto-restart) does
# not re-announce milestones that already fired.
#
# Install as a user service (survives logout if lingering is enabled):
#   systemctl --user daemon-reload
#   systemctl --user enable --now rl-sweep-watch
#   journalctl --user -u rl-sweep-watch -f
#
# Or just run it detached:
#   setsid nohup scripts/watch_sweep.sh >/dev/null 2>&1 &

set -uo pipefail

CONTAINER=sim
SWEEP_DIR=/checkpoints/sweep
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
STATE="$STATE_DIR/watch.state"
LOG="$STATE_DIR/watch.log"
POLL_SECONDS=120

mkdir -p "$STATE_DIR"
touch "$STATE"

log () { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }

# notify-send needs a session bus. When started by systemd --user these are
# usually inherited, but a bare `setsid nohup` from a subshell may not have
# them, so fall back to the conventional values for uid 1000's desktop.
export DISPLAY="${DISPLAY:-:1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

notify () {
  local urgency="$1" title="$2" body="$3"
  notify-send --urgency="$urgency" --app-name="rl_on_wheels" "$title" "$body" 2>/dev/null || true
  log "[$urgency] $title — $body"
}

fired () { grep -qxF "$1" "$STATE"; }
mark   () { echo "$1" >> "$STATE"; }

dexec () { docker exec "$CONTAINER" "$@" 2>/dev/null; }

container_up () { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }

read_json_field () {  # file, python expr over `d`
  dexec python3 -c "
import json
d=json.load(open('$1'))
print($2)
" 2>/dev/null
}

log "watcher started (pid $$)"

while true; do
  if ! container_up; then
    if ! fired "container_gone"; then
      mark "container_gone"
      notify critical "Sweep container stopped" \
        "The '$CONTAINER' container is not running. Any in-flight run has ended; checkpoints already written are still valid."
    fi
    sleep "$POLL_SECONDS"; continue
  fi

  # ── BC evaluations ──────────────────────────────────────────────────────────
  for mode in policy_alone with_overrides; do
    f="$SWEEP_DIR/bc_${mode}.json"
    if ! fired "bc_$mode" && dexec test -f "$f"; then
      mark "bc_$mode"
      rate=$(read_json_field "$f" "f\"{d['success_rate']:.1%} ({d['successes']}/{d['episodes']})\"")
      notify normal "BC eval done: $mode" "${rate:-complete}"
    fi
  done

  # ── TD3 seeds ───────────────────────────────────────────────────────────────
  for s in 0 1 2; do
    f="$SWEEP_DIR/td3_seed${s}/run_summary.json"
    if ! fired "seed_$s" && dexec test -f "$f"; then
      mark "seed_$s"
      info=$(read_json_field "$f" \
        "f\"{d['wall_total_h']:.1f} h, {len(d['budgets_reached'])}/{len(d['grid_bridge_steps'])} budgets, {d['mean_bridge_steps_per_sec']:.0f} steps/s\"")
      notify normal "TD3 sweep: seed $s complete" "${info:-done}"
    fi
  done

  # ── whole sweep ─────────────────────────────────────────────────────────────
  if ! fired "sweep_done" && dexec grep -q "all seeds done" /logs/sweep/orchestrator.log; then
    mark "sweep_done"
    notify critical "TD3 sweep FINISHED" \
      "All seeds done. Next: prepare_airl.sh, run_sweep_airl.sh, then eval_budget.py + analyze_crossover.py"
  fi

  # Stop once everything we care about has fired.
  if fired "sweep_done" && fired "bc_policy_alone" && fired "bc_with_overrides"; then
    log "all milestones reported; exiting"
    exit 0
  fi

  # ── failure detection ───────────────────────────────────────────────────────
  # Silence must not be mistaken for progress: if the trainer and orchestrator
  # are both gone without a completion marker, say so.
  if ! fired "sweep_done" && ! fired "sweep_died" \
     && ! dexec pgrep -f "run_sweep.sh|sweep_td3.py" >/dev/null; then
    mark "sweep_died"
    notify critical "TD3 sweep stopped early" \
      "No trainer or orchestrator running and no completion marker. Check /logs/sweep/orchestrator.log — budget checkpoints already written remain valid."
  fi

  sleep "$POLL_SECONDS"
done
