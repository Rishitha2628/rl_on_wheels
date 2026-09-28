#!/usr/bin/env bash
# Chain everything that follows the TD3 sweep, so the machine does not idle.
#
# Order is deliberate: the TD3 budget evaluations run FIRST, before the 25-hour
# AIRL arm. They take about 45 minutes and produce the paper's headline number
# -- where online RL crosses the imitation baseline. Knowing that before
# committing a night of compute is worth far more than the 45 minutes it
# delays AIRL by.
#
#   1. wait for TD3 seed 2 to finish
#   2. evaluate every TD3 budget checkpoint          (~45 min)
#   3. crossover figure + number, TD3 vs BC          (seconds)
#   4. rebuild the AIRL artifacts from the demos     (~15 min)
#   5. AIRL budget sweep, 3 seeds capped at 2M       (~25 h)
#
# Each step checks the previous one produced what it should, and stops rather
# than cascading a failure into a night of wasted compute.
#
# Run detached:
#   docker exec -d sim /scripts/run_after_sweep.sh

set -uo pipefail

SWEEP_DIR=/checkpoints/sweep
LOG=/logs/sweep/after_sweep.log
FIGDIR=/logs/figures
EPISODES=100
N_ENVS=4
STAGE=4
AIRL_ARTIFACTS=/checkpoints/airl_stage4

exec > >(stdbuf -oL tee -a "$LOG") 2>&1

say () { echo ""; echo "=== $(date '+%H:%M:%S')  $*"; }

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

# ── 1. wait for the sweep ─────────────────────────────────────────────────────
say "waiting for TD3 seed 2"
while pgrep -f "sweep_td3.py|run_sweep.sh" >/dev/null 2>&1; do sleep 60; done
if [[ ! -f "$SWEEP_DIR/td3_seed2/run_summary.json" ]]; then
  echo "seed 2 produced no run_summary.json; stopping rather than guessing." >&2
  exit 1
fi
say "TD3 sweep finished"

# ── 2. evaluate every TD3 budget checkpoint ───────────────────────────────────
say "launching fleet for evaluation"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
  echo "fleet failed to start" >&2; exit 1; }

eval_failed=0
for d in "$SWEEP_DIR"/td3_seed*; do
  [[ -f "$d/budget_index.json" ]] || { echo "skip $d (no budget_index.json)"; continue; }
  say "evaluating $(basename "$d")"
  stdbuf -oL python3 /rl/eval_budget.py \
      --run-dir "$d" --episodes "$EPISODES" --n-envs "$N_ENVS" \
    || { echo "eval failed for $d"; eval_failed=1; }
done

/scripts/launch_sim_fleet.sh --stop
sleep 5

# ── 3. crossover ──────────────────────────────────────────────────────────────
say "crossover analysis (TD3 vs BC)"
python3 /scripts/analyze_crossover.py \
    --sweep-dir "$SWEEP_DIR" \
    --bc-results "$SWEEP_DIR/bc_policy_alone.json" \
    --out-dir "$FIGDIR" || echo "analysis failed (checkpoint evals may be incomplete)"

say "HEADLINE NUMBERS ARE IN -> $FIGDIR/crossover.json and crossover.png"
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "TD3 crossover computed" "See /logs/figures/crossover.json" 2>/dev/null || true

if [[ $eval_failed -eq 1 ]]; then
  echo "One or more evaluations failed. Stopping before AIRL so the failure" >&2
  echo "can be looked at rather than buried under 25 hours of compute." >&2
  exit 1
fi

# ── 4. rebuild AIRL artifacts ─────────────────────────────────────────────────
say "rebuilding AIRL artifacts from the demos"
/scripts/prepare_airl.sh --out-dir "$AIRL_ARTIFACTS" || {
  echo "prepare_airl.sh failed; not starting the AIRL sweep." >&2; exit 1; }

for f in "$AIRL_ARTIFACTS/reward_net.pt" "$AIRL_ARTIFACTS/policy_after_bc.zip"; do
  [[ -f "$f" ]] || { echo "missing $f; not starting the AIRL sweep." >&2; exit 1; }
done

# ── 5. AIRL sweep ─────────────────────────────────────────────────────────────
say "starting the AIRL budget sweep (3 seeds, capped at 2M)"
/scripts/run_sweep_airl.sh --seeds "0 1 2" --max 2000000 \
    --artifacts "$AIRL_ARTIFACTS" --n-envs "$N_ENVS" --stage "$STAGE"

say "AIRL sweep finished — evaluate it with eval_budget.py, then re-run analyze_crossover.py"
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "AIRL sweep finished" "Run eval_budget.py on the airl_seed* dirs, then analyze_crossover.py" 2>/dev/null || true
