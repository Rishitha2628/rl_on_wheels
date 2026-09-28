#!/usr/bin/env bash
# Test whether the ~60% TD3 plateau is the reward's fault.
#
# Diagnosis. Per bridge step the shaping terms are roughly:
#   r_yaw      -3.14 .. 0
#   r_vangular -4.00 .. 0   (-w * omega^2, omega up to 2.0)
#   r_vlinear  -4.84 .. 0   (-((0.22-v)*scale)^2, worst at v=0)
#   constant   -1
# A robot that slows and turns -- exactly what obstacle avoidance requires --
# therefore accrues about -13 per step, or about -5850 over a 450-step episode,
# against a collision penalty of only -2000. Slowing down costs roughly three
# times more than crashing, so the policy learns to drive fast and straight.
# That matches what we measured: as success rose, mean episode length halved.
#
# This project hit ~59% previously and ~64% in the sweep, from different
# hyperparameters. A shared reward defect explains both better than tuning does.
#
# Both variants use the from-scratch-appropriate optimiser settings (lr 3e-4,
# OU sigma 0.1) so exploration is not simultaneously the bottleneck.
#
# Success is scored by goal-reaching at 0.40 m, which does not involve the
# reward at all, so these runs stay directly comparable with BC and with the
# baseline TD3 curve.
#
#   docker exec -d sim /scripts/run_reward_variants.sh

set -uo pipefail

CONF=/configs/td3_tune_a.yaml     # lr 3e-4, sigma 0.1
OUT_ROOT=/checkpoints/tune
LOG=/logs/sweep/reward_variants.log
BUDGET=640000
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

say "waiting for the tuning diagnostic to finish"
while pgrep -f "finish_and_tune.sh" >/dev/null 2>&1; do sleep 60; done

# ── rebuild ───────────────────────────────────────────────────────────────────
# Deferred until now on purpose: rebuilding mid-diagnostic would have swapped
# the binary between tuning runs A and B. touch first -- the image's object
# files can carry a later mtime than a host edit, and make will silently skip.
say "rebuilding tb3_rl_bridge with the reward weights exposed"
touch /ros2_ws/src/tb3_rl_bridge/src/*.cpp
cd /ros2_ws && colcon build --symlink-install --packages-select tb3_rl_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cd /

# NOTE: written as a command substitution, not `if ! strings ... | grep -q`.
# Under `set -o pipefail`, grep -q exits on its first match and SIGPIPEs
# strings, so the pipeline returns 141 and the negated test fires even though
# the rebuild succeeded. That exact bug aborted this script on 2026-08-28 --
# a check added to catch a silent failure caused one instead.
n_params=$(strings /ros2_ws/build/tb3_rl_bridge/env_bridge_node 2>/dev/null \
             | grep -c r_vlinear_scale || true)
if [[ "${n_params:-0}" -lt 1 ]]; then
  echo "rebuild did not pick up the new parameters; stopping." >&2
  exit 1
fi
say "rebuild verified: r_vlinear_scale present in the binary"

# ── variants ──────────────────────────────────────────────────────────────────
# c: moderate  -- forward penalty 4x smaller, turning penalty halved
# d: aggressive-- forward penalty 16x smaller, turning penalty quartered
run_variant () {
  local name="$1" vlin="$2" vang="$3"
  local out="$OUT_ROOT/td3_reward_${name}"

  say "variant ${name}: r_vlinear_scale=${vlin} r_vangular_weight=${vang}"
  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast \
      --r-vlinear "$vlin" --r-vangular "$vang" || {
    echo "fleet failed for variant ${name}" >&2; return 1; }

  stdbuf -oL python3 /rl/sweep_td3.py \
      --config "$CONF" --seed 0 --n-envs "$N_ENVS" \
      --out-dir "$out" --max-bridge-steps "$BUDGET" 2>&1 | tail -30

  say "evaluating variant ${name}"
  stdbuf -oL python3 /rl/eval_budget.py --run-dir "$out" --config "$CONF" \
      --episodes "$EPISODES" --n-envs "$N_ENVS" || echo "eval failed for ${name}"

  /scripts/launch_sim_fleet.sh --stop
  sleep 5
}

run_variant c 5.0 0.5
run_variant d 2.5 0.25

# ── verdict ───────────────────────────────────────────────────────────────────
say "REWARD VARIANTS COMPLETE"
python3 - <<'PY'
import json, os

def best(p):
    f = os.path.join(p, "eval_results.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))["results"]
    return max(r, key=lambda x: x["success_rate"]) if r else None

rows = [
    ("baseline  lr1e-4 s0.05 rew10/1.0", "/checkpoints/sweep/td3_seed1"),
    ("tuned  a  lr3e-4 s0.10 rew10/1.0", "/checkpoints/tune/td3_tune_a"),
    ("tuned  b  lr3e-4 s0.20 rew10/1.0", "/checkpoints/tune/td3_tune_b"),
    ("reward c  lr3e-4 s0.10 rew5/0.5",  "/checkpoints/tune/td3_reward_c"),
    ("reward d  lr3e-4 s0.10 rew2.5/0.25", "/checkpoints/tune/td3_reward_d"),
]
print("\n  %-36s %8s   %-14s" % ("configuration", "best", "95% CI"))
print("  " + "-"*64)
for label, path in rows:
    b = best(path)
    if b:
        print("  %-36s %7.1f%%   [%.0f, %.0f]" % (
            label, b["success_rate"]*100,
            b["success_ci95"][0]*100, b["success_ci95"][1]*100))
    else:
        print("  %-36s %8s" % (label, "--"))
print("\n  BC policy alone 63.7%% [58.1, 68.9]   BC + overrides 78.0%% [73.0, 82.3]")
print("\n  If a reward variant clears ~70%, the plateau was the reward's and the")
print("  TD3 arm should be re-run with that shaping. If all land near 64%, the")
print("  plateau is real and the paper's conclusion is on firm ground.")
PY

notify-send --urgency=critical --app-name="rl_on_wheels" \
  "Reward variants done" "See /logs/sweep/reward_variants.log" 2>/dev/null || true
