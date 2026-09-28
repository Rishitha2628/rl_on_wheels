#!/usr/bin/env bash
# Evaluate the AIRL arm and regenerate the three-arm crossover figure.
#
# run_after_sweep.sh deliberately ends when the AIRL sweep does, so a failure
# there could be inspected rather than buried. But that leaves a manual step
# between the last seed finishing and the figure existing. This closes it:
# wait for the sweep, evaluate every AIRL budget checkpoint, then re-run the
# analysis over all three arms.
#
# Safe to start now -- it waits rather than assuming the sweep is done.
#
#   docker exec -d sim /scripts/finish_airl.sh

set -uo pipefail

SWEEP_DIR=/checkpoints/sweep
FIGDIR=/logs/figures
LOG=/logs/sweep/finish_airl.log
EPISODES=100
N_ENVS=4
STAGE=4

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%H:%M:%S')  $*"; }

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

say "waiting for the AIRL sweep to finish"
while pgrep -f "sweep_airl.py|run_sweep_airl.sh" >/dev/null 2>&1; do sleep 60; done

done_seeds=0
for d in "$SWEEP_DIR"/airl_seed*; do
  [[ -f "$d/run_summary.json" ]] && done_seeds=$((done_seeds + 1))
done
if (( done_seeds == 0 )); then
  echo "No AIRL run summaries found; stopping rather than producing an empty figure." >&2
  exit 1
fi
say "AIRL sweep finished ($done_seeds seeds)"

say "launching fleet for evaluation"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
  echo "fleet failed to start" >&2; exit 1; }

failed=0
for d in "$SWEEP_DIR"/airl_seed*; do
  [[ -f "$d/budget_index.json" ]] || { echo "skip $d (no budget_index.json)"; continue; }
  say "evaluating $(basename "$d")"
  stdbuf -oL python3 /rl/eval_budget.py \
      --run-dir "$d" --episodes "$EPISODES" --n-envs "$N_ENVS" \
    || { echo "eval failed for $d"; failed=1; }
done

/scripts/launch_sim_fleet.sh --stop
sleep 5

say "regenerating the crossover figure across all three arms"
python3 /scripts/analyze_crossover.py \
    --sweep-dir "$SWEEP_DIR" \
    --bc-results "$SWEEP_DIR/bc_policy_alone.json" \
    --out-dir "$FIGDIR"

say "ALL ARMS COLLECTED -> $FIGDIR/crossover.{json,png,pdf}"
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "All three arms collected" \
  "Crossover figure regenerated with BC, TD3 and AIRL. See /logs/figures/" 2>/dev/null || true

(( failed == 1 )) && echo "NOTE: at least one evaluation failed; the figure may be incomplete." >&2
exit 0
