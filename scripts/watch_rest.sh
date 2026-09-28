#!/usr/bin/env bash
# Watches everything still queued: the reward variants and then the AIRL
# re-run. Exits only when the last of them finishes.
#
# Each earlier watcher covered one stage and exited when that stage's
# milestones had fired -- correct in isolation, but it meant the durable
# notification channel lapsed exactly as the next stage began, four times over.
# This one is scoped to the remainder of the work.
#
#   systemctl --user enable --now rl-rest-watch

set -uo pipefail

CONTAINER=sim
RV_LOG=/logs/sweep/reward_variants.log
AIRL_LOG=/logs/sweep/rerun_airl.log
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
STATE="$STATE_DIR/rest_watch.state"
LOG="$STATE_DIR/rest_watch.log"
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

best_of () {  # dir -> "62.0% at 640,000 steps"
  dexec python3 -c "
import json
r = json.load(open('$1/eval_results.json'))['results']
b = max(r, key=lambda x: x['success_rate'])
print('%.1f%% [%.0f-%.0f] at %s steps' % (
    b['success_rate']*100, b['success_ci95'][0]*100,
    b['success_ci95'][1]*100, f\"{b['budget_bridge_steps']:,}\"))
" 2>/dev/null
}

log "rest watcher started (pid $$)"

while true; do
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    if ! fired container_gone; then
      mark container_gone
      notify critical "Sweep container stopped" "Results already written remain valid."
    fi
    sleep "$POLL"; continue
  fi

  # ── reward variants ─────────────────────────────────────────────────────────
  if ! fired rebuild && dexec grep -q "rebuild verified" "$RV_LOG"; then
    mark rebuild
    notify normal "Reward weights built" "Starting variant c (4x weaker forward penalty)."
  fi

  for v in c d; do
    if ! fired "rv_$v" && dexec test -f "/checkpoints/tune/td3_reward_${v}/eval_results.json"; then
      mark "rv_$v"
      notify normal "Reward variant ${v} evaluated" \
        "$(best_of /checkpoints/tune/td3_reward_${v})  (baseline 64.5%, BC 63.7%)"
    fi
  done

  if ! fired rv_done && dexec grep -q "REWARD VARIANTS COMPLETE" "$RV_LOG"; then
    mark rv_done
    notify critical "Reward variants done" \
      "Verdict table in $RV_LOG — did softening the speed penalty break the 60% plateau?"
  fi

  # ── AIRL re-run ─────────────────────────────────────────────────────────────
  if ! fired airl_peak && dexec grep -q "AIRL generator peaked" "$AIRL_LOG"; then
    mark airl_peak
    peak=$(dexec grep -o "AIRL generator peaked at .*" "$AIRL_LOG" | tail -1)
    notify normal "AIRL phase 1 done" "${peak:-see log} (needs >= 0.30 to continue)"
  fi

  if ! fired airl_done && dexec grep -q "AIRL RE-RUN COMPLETE" "$AIRL_LOG"; then
    mark airl_done
    notify critical "AIRL re-run complete" "Three-arm figure redrawn. /logs/figures/"
    log "all milestones reported; exiting"
    exit 0
  fi

  if ! fired airl_skipped && dexec grep -q "peak below 0.30" "$AIRL_LOG"; then
    mark airl_skipped
    notify critical "AIRL re-run abandoned" \
      "Generator collapsed again; phase 2 skipped deliberately. Report as a negative result."
    exit 0
  fi

  # Silence must not read as success.
  if ! fired all_gone \
     && ! dexec pgrep -f "run_reward_variants.sh|rerun_airl.sh|finish_and_tune.sh" >/dev/null; then
    mark all_gone
    notify critical "Queued work stopped early" \
      "No chain running and no completion marker. Check $RV_LOG and $AIRL_LOG."
    exit 0
  fi

  sleep "$POLL"
done
