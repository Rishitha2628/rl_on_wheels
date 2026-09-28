#!/usr/bin/env bash
# Evaluate the BC policy to produce the sweep's horizontal reference line.
#
# Runs TWO evaluations, because they answer different questions:
#
#   policy alone (--no-overrides)   the number that belongs on the crossover
#                                   plot. TD3 and AIRL run with no hand-written
#                                   help, so the baseline must not either.
#
#   with overrides                  what the BC pipeline actually ships: safety
#                                   guards, a speed cap and a near-goal
#                                   P-controller wrapped around the policy.
#                                   Worth reporting -- it is the honest number
#                                   for "what BC delivers" -- but it is not a
#                                   like-for-like comparison against a bare
#                                   learned policy.
#
# Single instance rather than a fleet: bc/eval.py drives one env directly and
# a solo instance runs at ~58 steps/s versus ~27 under 4-way contention.
#
# Usage:
#   scripts/eval_bc.sh                        # 50 episodes, both modes
#   scripts/eval_bc.sh --episodes 100

set -uo pipefail

EPISODES=50
STAGE=4
CHECKPOINT=/checkpoints/bc_best.pt
CONFIG=/configs/bc_stage4.yaml
OUT_DIR=/checkpoints/sweep
# Well clear of the sweep's 40-43 so the two never share a DDS graph.
EVAL_DOMAIN=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)   EPISODES="$2"; shift 2 ;;
    --stage)      STAGE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    -h|--help)    sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SWEEP_RUNNING=0
if pgrep -f "sweep_td3.py|sweep_airl.py" >/dev/null 2>&1; then
  SWEEP_RUNNING=1
  echo "[bc-eval] A sweep is running. This evaluation will share the machine."
  echo "[bc-eval]   - It uses its own sim instance on ROS_DOMAIN_ID=$EVAL_DOMAIN,"
  echo "[bc-eval]     isolated from the sweep's fleet on 40-43, and never calls"
  echo "[bc-eval]     the fleet teardown (which does a global pkill and would"
  echo "[bc-eval]     kill the sweep)."
  echo "[bc-eval]   - Robot-hours in the paper are derived from step counts, not"
  echo "[bc-eval]     from wall clock, so contention does not change any"
  echo "[bc-eval]     reported result. It only makes the sweep take longer."
  echo "[bc-eval]   - Cost: a few minutes of ~20% slower sweep."
  echo ""
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[bc-eval] MISSING $CHECKPOINT" >&2
  echo "[bc-eval] Train it first (offline, no sim needed):" >&2
  echo "    python3 -m bc.train --config $CONFIG \\" >&2
  echo "        --dataset /demos/stage4_dagger_agg2.npz" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

mkdir -p "$OUT_DIR" /logs/sweep

# ── isolated sim instance ─────────────────────────────────────────────────────
# Launched inline rather than through launch_sim_fleet.sh on purpose. That
# script begins by tearing down every running sim (a global `pkill -9 -f "ign
# gazebo"`), which would kill a sweep mid-flight. Nothing here touches any
# process it did not start.
EVAL_PARTITION=tb3eval

cleanup_eval_sim () {
  # Kill our own remaining children first (the running bc/eval.py). They sit in
  # this script's process group, not the sim's, so signalling only the sim
  # group leaves an orphaned evaluator holding the domain.
  pkill -TERM -P $$ 2>/dev/null
  if [[ -n "${EVAL_PGID:-}" ]]; then
    kill -TERM "-$EVAL_PGID" 2>/dev/null
    sleep 3
    kill -9 "-$EVAL_PGID" 2>/dev/null
  fi
  pkill -9 -P $$ 2>/dev/null
}
trap cleanup_eval_sim EXIT INT TERM

echo "[bc-eval] starting an isolated sim on ROS_DOMAIN_ID=$EVAL_DOMAIN "\
"IGN_PARTITION=$EVAL_PARTITION"
setsid env \
  ROS_DOMAIN_ID="$EVAL_DOMAIN" \
  IGN_PARTITION="$EVAL_PARTITION" \
  ros2 launch tb3_rl_bridge bridge.launch.py \
    stage:="$STAGE" headless:=true fast:=true \
  > /logs/sweep/bc_eval_sim.log 2>&1 &
EVAL_PGID=$!

echo "[bc-eval] waiting for /step on domain $EVAL_DOMAIN ..."
ready=0
for _ in $(seq 1 40); do
  if ROS_DOMAIN_ID="$EVAL_DOMAIN" timeout 8 ros2 service list 2>/dev/null \
       | grep -q '^/step$'; then
    ready=1; break
  fi
  sleep 3
done
if [[ $ready -eq 0 ]]; then
  echo "[bc-eval] sim did not come up; see /logs/sweep/bc_eval_sim.log" >&2
  exit 1
fi
echo "[bc-eval] sim ready."

run_eval () {
  local label="$1"; shift
  echo ""
  echo "=================================================================="
  echo "[bc-eval] $label  ($EPISODES episodes)"
  echo "=================================================================="
  ROS_DOMAIN_ID="$EVAL_DOMAIN" IGN_PARTITION="$EVAL_PARTITION" \
    stdbuf -oL -eL python3 /bc/eval.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --episodes "$EPISODES" \
      --out "$OUT_DIR/bc_${label}.json" \
      --quiet "$@" 2>&1 | tee "/logs/sweep/bc_${label}.log"
}

run_eval policy_alone --no-overrides
run_eval with_overrides

# No fleet teardown here — the EXIT trap kills only this script's own sim
# process group. Calling launch_sim_fleet.sh --stop would pkill every sim on
# the machine, including a running sweep's.
cleanup_eval_sim
trap - EXIT INT TERM

echo ""
echo "[bc-eval] results:"
for f in "$OUT_DIR"/bc_policy_alone.json "$OUT_DIR"/bc_with_overrides.json; do
  [[ -f "$f" ]] && python3 -c "
import json
d = json.load(open('$f'))
print(f\"  {'overrides ON ' if d['overrides_enabled'] else 'policy alone'}: \"
      f\"{d['success_rate']:.1%} ({d['successes']}/{d['episodes']})  \"
      f\"reward {d['mean_reward']:.0f}  len {d['mean_length']:.0f}\")
"
done
echo ""
echo "[bc-eval] use bc_policy_alone.json as --bc-results for analyze_crossover.py"
