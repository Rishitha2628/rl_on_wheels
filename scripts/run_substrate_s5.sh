#!/usr/bin/env bash
# The substrate factorial and the evaluation-wrapper measurement, on STAGE 5
# (7x7 arena, 6 m lidar, 6 moving cylinders) -- the arena the original BC/AIRL/
# TD3 work was done on. Mirrors run_substrate.sh, which does the same on stage 4.
#
# Policies come from /demos/stage11_dagger_agg1.npz (468 episodes, 90,144
# transitions), one DAgger round fewer than the stage-4 set.
#
#   docker exec -d sim /scripts/run_substrate_s5.sh
#
# N_ENVS defaults to 2, not 4: the box ran at 93 C on the last long job, and
# two instances give ~73% of four-instance throughput for much less heat.
# Pairing only holds across runs that used the SAME N_ENVS, because instance i
# draws goal seed GOAL_SEED+i.

set -uo pipefail

GOAL_SEED=1000
EPISODES=300
N_ENVS=${N_ENVS:-2}
STAGE=5
OUT=/checkpoints/substrate_s5
LOG=/logs/sweep/substrate_s5.log
MIN_FREE_GB=2

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; }

check_room () {
  local free
  free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  echo "  [disk] ${free}G free"
  (( free < MIN_FREE_GB )) && { echo "  [disk] below ${MIN_FREE_GB}G, stopping." >&2; return 1; }
  return 0
}

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

if ps -eo args | grep -q '^python3 /rl/sweep_td3.py'; then
  echo "A trainer is already running; refusing to start a second." >&2
  exit 1
fi

say "STAGE 5 SUBSTRATE RUNS START  (goal seeds ${GOAL_SEED}..$((GOAL_SEED + N_ENVS - 1)), N_ENVS=${N_ENVS})"
check_room || exit 1

for fs in 1 4; do
  [[ -f "/checkpoints/s5_fs${fs}/bc_best.pt" ]] || {
    echo "missing /checkpoints/s5_fs${fs}/bc_best.pt -- train it first:" >&2
    echo "  python3 -m bc.train --config /configs/bc_s5_fs${fs}.yaml \\" >&2
    echo "      --dataset /demos/stage11_dagger_agg1.npz" >&2
    exit 1
  }
done

say "launching stage-5 fleet on fixed goals"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast --goal-seed "$GOAL_SEED" || exit 1

# ── the 2x2 factorial: frame_stack x control rate, no overrides ───────────────
for fs in 1 4; do
  for ar in 1 4; do
    res="$OUT/bc_fs${fs}_ar${ar}.json"
    [[ -f "$res" ]] && { echo "  fs=$fs ar=$ar already evaluated"; continue; }
    say "BC frame_stack=${fs} action_repeat=${ar}"
    check_room || break 2
    # ROS_DOMAIN_ID, not the goal seed: domain IDs must be 0-232 and the fleet
    # puts instance 0 on DOMAIN_BASE=40.
    ROS_DOMAIN_ID=40 python3 /bc/eval.py \
        --config "/configs/bc_s5_fs${fs}.yaml" \
        --checkpoint "/checkpoints/s5_fs${fs}/bc_best.pt" \
        --episodes "$EPISODES" --action-repeat "$ar" \
        --no-overrides --quiet --out "$res" 2>&1 | tail -4
    [[ -f "$res" ]] || { echo "  no output for fs=$fs ar=$ar; stopping rather than" >&2
                         echo "  reporting a factorial with missing cells." >&2; exit 1; }
  done
done

# ── the evaluation wrapper, at BC's native setting (fs=4, 10 Hz) ──────────────
for mode in with_overrides policy_alone; do
  res="$OUT/bc_${mode}.json"
  [[ -f "$res" ]] && { echo "  $mode already evaluated"; continue; }
  say "BC fs=4 ar=1, ${mode}"
  check_room || break
  flag=""; [[ "$mode" == "policy_alone" ]] && flag="--no-overrides"
  ROS_DOMAIN_ID=40 python3 /bc/eval.py \
      --config /configs/bc_s5_fs4.yaml \
      --checkpoint /checkpoints/s5_fs4/bc_best.pt \
      --episodes "$EPISODES" --action-repeat 1 \
      $flag --quiet --out "$res" 2>&1 | tail -4
done

/scripts/launch_sim_fleet.sh --stop; sleep 5

say "STAGE 5 SUBSTRATE RUNS COMPLETE"
for f in "$OUT"/*.json; do
  [[ -f "$f" ]] && python3 -c "
import json,sys
d=json.load(open('$f')); import os
print(f\"  {os.path.basename('$f'):26s} {d['successes']:>3d}/{d['episodes']} = {100*d['success_rate']:.1f}%\")"
done
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "Stage-5 substrate done" "See $LOG" 2>/dev/null || true
