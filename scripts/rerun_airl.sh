#!/usr/bin/env bash
# Re-run the AIRL arm with the reward net taken from the best generator round
# rather than the last one.
#
# The first attempt produced ~1% success. The cause was not the learning rate
# (train_airl.py correctly uses 3e-5) but WHEN the reward net was saved: the
# adversarial loop degraded the generator from 0.50 to 0.10 across rounds, and
# the reward net written at the end was the one fitted against the collapsed
# policy. Phase 2 then faithfully optimised a reward learned by watching a bad
# agent.
#
# train_airl.py now snapshots the reward net alongside the best-eval policy,
# and prepare_airl.sh promotes that snapshot. --eval-every 5 rather than 10
# doubles the number of snapshot opportunities; evaluations are 10 episodes, so
# the extra cost is negligible against a 25-hour arm.
#
# Waits for the tuning diagnostic first -- that result decides whether the TD3
# arm (the paper's spine) is sound, and it should not be slowed down.
#
#   docker exec -d sim /scripts/rerun_airl.sh

set -uo pipefail

ARTIFACTS=/checkpoints/airl_stage4_v2
SWEEP_DIR=/checkpoints/sweep
FIGDIR=/logs/figures
LOG=/logs/sweep/rerun_airl.log
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

# Waits for BOTH the tuning diagnostic and the reward-variant runs. TD3 is the
# paper's spine and AIRL is a supporting arm, so the TD3 questions get the
# machine first.
say "waiting for the tuning diagnostic and reward variants to finish"
while pgrep -f "finish_and_tune.sh|run_reward_variants.sh" >/dev/null 2>&1; do
  sleep 120
done
say "TD3 work finished; starting the AIRL re-run"

# ── 1. rebuild reward + warm start, snapshotting at best eval ─────────────────
say "train_airl.py with --eval-every 5"
/scripts/prepare_airl.sh --out-dir "$ARTIFACTS" --airl-steps 60000 \
    --seed 0 --stage "$STAGE" || {
  echo "prepare_airl.sh failed; stopping." >&2; exit 1; }

for f in "$ARTIFACTS/reward_net.pt" "$ARTIFACTS/policy_after_bc.zip"; do
  [[ -f "$f" ]] || { echo "missing $f; stopping." >&2; exit 1; }
done

# Refuse to proceed if the generator never got anywhere -- a reward net from a
# run that peaked at 10% is not worth 25 hours of phase-2 compute.
peak=$(python3 -c "
import yaml
try:
    print(yaml.safe_load(open('$ARTIFACTS/eval_log.yaml'))['best_success'])
except Exception:
    print(0)
" 2>/dev/null)
say "AIRL generator peaked at ${peak}"
if python3 -c "import sys; sys.exit(0 if float('$peak') >= 0.30 else 1)"; then
  echo "  peak is usable; continuing to phase 2."
else
  echo "  peak below 0.30 -- the adversarial loop failed again. Not spending" >&2
  echo "  25 hours optimising a reward learned from a policy this poor." >&2
  notify-send --urgency=critical --app-name="rl_on_wheels" \
    "AIRL re-run failed again" "Generator peaked at ${peak}; phase 2 skipped. Report as a negative result." 2>/dev/null || true
  exit 1
fi

# ── 2. phase-2 budget sweep ───────────────────────────────────────────────────
say "AIRL budget sweep, 3 seeds capped at 2M"
/scripts/run_sweep_airl.sh --seeds "0 1 2" --max 2000000 \
    --artifacts "$ARTIFACTS" --n-envs "$N_ENVS" --stage "$STAGE" \
    --out "$SWEEP_DIR"

# ── 3. evaluate and redraw ────────────────────────────────────────────────────
say "evaluating the AIRL arm"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
  echo "fleet failed" >&2; exit 1; }
for d in "$SWEEP_DIR"/airl_seed*; do
  [[ -f "$d/budget_index.json" ]] || continue
  stdbuf -oL python3 /rl/eval_budget.py \
      --run-dir "$d" --episodes "$EPISODES" --n-envs "$N_ENVS" \
    || echo "eval failed for $d"
done
/scripts/launch_sim_fleet.sh --stop
sleep 5

say "redrawing the three-arm figure"
python3 /scripts/analyze_crossover.py --sweep-dir "$SWEEP_DIR" \
    --bc-results "$SWEEP_DIR/bc_policy_alone.json" --out-dir "$FIGDIR"

say "AIRL RE-RUN COMPLETE"
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "AIRL re-run complete" "Three-arm figure redrawn. See /logs/figures/" 2>/dev/null || true
