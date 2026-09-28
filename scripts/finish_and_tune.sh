#!/usr/bin/env bash
# Re-run the AIRL evaluations (they failed the first time), regenerate the
# three-arm figure, then run the TD3 tuning diagnostic.
#
# The AIRL evals failed because eval_budget.py assumed the TD3 arm: it loaded
# checkpoints with TD3.load over the raw 42-dim observation, while AIRL
# policies are PPO over the 40-dim Strip42to40 space. eval_budget.py now picks
# the algorithm and env from the run's own run_meta.json.
#
# The tuning runs exist because configs/td3.yaml's own comments say its
# learning rate (1e-4) and exploration noise (sigma 0.05) were chosen to
# FINE-TUNE an already-converged policy that had plateaued at ~0.59 -- and the
# sweep trained from scratch with them, measuring 0.62-0.645. That is close
# enough to the documented plateau to suspect the result is an artifact of the
# configuration rather than a property of the task. Two short runs at
# from-scratch-appropriate values settle it.
#
#   docker exec -d sim /scripts/finish_and_tune.sh

set -uo pipefail

SWEEP_DIR=/checkpoints/sweep
TUNE_DIR=/checkpoints/tune
FIGDIR=/logs/figures
LOG=/logs/sweep/finish_and_tune.log
EPISODES=100
N_ENVS=4
STAGE=4
TUNE_BUDGET=640000     # the TD3 curve is flat from 320k; 640k is enough to see it

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%H:%M:%S')  $*"; }

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

mkdir -p "$TUNE_DIR"

/scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1
sleep 3

# ── 1. AIRL evaluations ───────────────────────────────────────────────────────
say "launching fleet"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
  echo "fleet failed to start" >&2; exit 1; }

airl_failed=0
for d in "$SWEEP_DIR"/airl_seed*; do
  [[ -f "$d/budget_index.json" ]] || continue
  say "evaluating $(basename "$d")"
  stdbuf -oL python3 /rl/eval_budget.py \
      --run-dir "$d" --episodes "$EPISODES" --n-envs "$N_ENVS" \
    || { echo "eval failed for $d"; airl_failed=1; }
done

say "regenerating the three-arm figure"
python3 /scripts/analyze_crossover.py \
    --sweep-dir "$SWEEP_DIR" \
    --bc-results "$SWEEP_DIR/bc_policy_alone.json" \
    --out-dir "$FIGDIR"

notify-send --urgency=critical --app-name="rl_on_wheels" \
  "Three-arm figure ready" "BC, TD3 and AIRL. See /logs/figures/" 2>/dev/null || true

/scripts/launch_sim_fleet.sh --stop
sleep 5

# ── 2. TD3 tuning diagnostic ──────────────────────────────────────────────────
# Is ~64% the task's ceiling, or this configuration's?
for cfg in a b; do
  conf="/configs/td3_tune_${cfg}.yaml"
  out="$TUNE_DIR/td3_tune_${cfg}"
  lr=$(grep -m1 '^  learning_rate:' "$conf" | awk '{print $2}')
  sg=$(grep -m1 '^    sigma:' "$conf" | awk '{print $2}')

  say "tuning run ${cfg}: lr=${lr} sigma=${sg}, ${TUNE_BUDGET} bridge steps"
  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
    echo "fleet failed for tuning ${cfg}" >&2; continue; }

  stdbuf -oL python3 /rl/sweep_td3.py \
      --config "$conf" --seed 0 --n-envs "$N_ENVS" \
      --out-dir "$out" --max-bridge-steps "$TUNE_BUDGET" \
    2>&1 | tail -40

  say "evaluating tuning run ${cfg}"
  stdbuf -oL python3 /rl/eval_budget.py \
      --run-dir "$out" --config "$conf" \
      --episodes "$EPISODES" --n-envs "$N_ENVS" || echo "eval failed for $out"

  /scripts/launch_sim_fleet.sh --stop
  sleep 5
done

# ── 3. verdict ────────────────────────────────────────────────────────────────
say "TUNING DIAGNOSTIC COMPLETE"
python3 - <<'PY'
import json, os, glob
base = "/checkpoints/sweep"
def best(path):
    f = os.path.join(path, "eval_results.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))["results"]
    return max(r, key=lambda x: x["success_rate"]) if r else None

print("\n  config                     best success   at budget   95% CI")
print("  " + "-"*62)
for seed in (0, 1, 2):
    b = best(f"{base}/td3_seed{seed}")
    if b:
        print("  baseline lr1e-4 s0.05 (s%d)  %11.1f%% %11s   [%.0f, %.0f]" % (
            seed, b["success_rate"]*100, f'{b["budget_bridge_steps"]:,}',
            b["success_ci95"][0]*100, b["success_ci95"][1]*100))
for cfg in ("a", "b"):
    b = best(f"/checkpoints/tune/td3_tune_{cfg}")
    if b:
        print("  tuned %s                    %11.1f%% %11s   [%.0f, %.0f]" % (
            cfg, b["success_rate"]*100, f'{b["budget_bridge_steps"]:,}',
            b["success_ci95"][0]*100, b["success_ci95"][1]*100))
print("\n  BC policy alone: 63.7%% [58.1, 68.9]   BC + overrides: 78.0%% [73.0, 82.3]")
print("\n  If the tuned runs also land near 64%, the plateau is the task's and")
print("  the paper's conclusion holds. If one clears ~70%, the baseline was")
print("  undertuned and the conclusion needs revising.")
PY

notify-send --urgency=critical --app-name="rl_on_wheels" \
  "Tuning diagnostic done" "See /logs/sweep/finish_and_tune.log" 2>/dev/null || true

(( airl_failed == 1 )) && echo "NOTE: an AIRL eval failed; figure may be incomplete." >&2
exit 0
