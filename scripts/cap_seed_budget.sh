#!/usr/bin/env bash
# Cap seeds 1 and 2 of the TD3 sweep at 2M bridge steps, leaving seed 0 to run
# the full 4M.
#
# Why this shape. run_sweep.sh is already executing with --max 4000000 baked
# into its loop, and bash reads a script lazily by byte offset -- editing it
# mid-flight makes the running shell resume at the wrong place. So instead of
# changing the orchestrator, this watches for seeds 1 and 2 and sends SIGINT
# once they pass the cap. sweep_td3.py catches KeyboardInterrupt and runs its
# `finally` block, which writes run_summary.json and the final model, so the
# run ends cleanly and the orchestrator moves on to the next seed by itself.
#
# SIGINT, never SIGKILL: -9 skips the finally block and the run summary would
# never be written.
#
# Rationale for the cap: 2M bridge steps is 55.9 robot-hours, already 20-50x
# any published real-robot training budget. Seed 0 continues to 4M so the
# plateau is still demonstrated out to 112 robot-hours, just with one seed.

set -uo pipefail

CONTAINER=sim
CAP_BRIDGE=2000000
ACTION_REPEAT=4
CAPPED_SEEDS="1 2"
POLL=60

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/rl_on_wheels"
LOG="$STATE_DIR/cap.log"
mkdir -p "$STATE_DIR"

CAP_AGENT=$((CAP_BRIDGE / ACTION_REPEAT))

log () { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
log "cap watcher started (pid $$): seeds [$CAPPED_SEEDS] capped at ${CAP_BRIDGE} bridge steps (${CAP_AGENT} agent steps)"

already=""

while true; do
  # Stop once the orchestrator is gone -- nothing left to cap.
  if ! docker exec "$CONTAINER" pgrep -f "run_sweep.sh" >/dev/null 2>&1; then
    if ! docker exec "$CONTAINER" pgrep -f "sweep_td3.py" >/dev/null 2>&1; then
      log "orchestrator and trainer both gone; exiting"
      exit 0
    fi
  fi

  pid=$(docker exec "$CONTAINER" pgrep -f "sweep_td3.py" 2>/dev/null | head -1)
  if [[ -n "$pid" ]]; then
    seed=$(docker exec "$CONTAINER" bash -c \
      "tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 -x -- '--seed' | tail -1" 2>/dev/null)

    if [[ " $CAPPED_SEEDS " == *" $seed "* ]] && [[ "$already" != *"[$seed]"* ]]; then
      steps=$(docker exec "$CONTAINER" bash -c \
        "grep -E '^\|    total_timesteps' /logs/sweep/td3_seed${seed}.log 2>/dev/null | tail -1 | awk '{print \$4}'" 2>/dev/null)

      if [[ "$steps" =~ ^[0-9]+$ ]] && (( steps >= CAP_AGENT )); then
        log "seed $seed reached $steps agent steps (>= $CAP_AGENT); sending SIGINT to pid $pid"
        docker exec "$CONTAINER" kill -INT "$pid" 2>/dev/null
        already="$already[$seed]"
        notify-send --urgency=normal --app-name="rl_on_wheels" \
          "TD3 seed $seed capped at 2M" \
          "Reached the 55.9 robot-hour cap; stopping cleanly so the next seed can start." 2>/dev/null || true
      fi
    fi
  fi

  sleep "$POLL"
done
