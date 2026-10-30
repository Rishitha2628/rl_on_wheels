#!/usr/bin/env bash
# Two more training seeds for the 1-frame and 4-frame policies on the 7x7 arena,
# then evaluate all four.
#
# The frame-stacking result rests on comparing two separately trained policies,
# one evaluation each. McNemar handles episode noise; it says nothing about
# training noise. Since we argue elsewhere that one seed is not a measurement,
# the same standard has to apply to our own strongest surviving claim.
#
# Also re-renders the stage-5 arena figure, which currently shows stage 4.
set -uo pipefail
LOG=/logs/sweep/bc_seeds.log
STATUS=/logs/sweep/bc_seeds_status.txt
GOAL_SEED=1000; EPISODES=300
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p /checkpoints/bc_seeds /logs/sweep

say "training 2 extra seeds x 2 frame-stack settings"
for sd in 1 2; do for fs in 1 4; do
  d=/checkpoints/s5d_fs${fs}_seed${sd}
  [[ -f "$d/bc_best.pt" ]] && { echo "  (exists) fs=$fs seed=$sd"; continue; }
  echo "  training fs=$fs seed=$sd"
  python3 -m bc.train --config /configs/bc_s5d_fs${fs}_seed${sd}.yaml \
      --dataset /demos/stage11_dagger_agg2.npz --seed "$sd" 2>&1 | tail -1
done; done

say "evaluating"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 1 2; do for fs in 1 4; do
  out=/checkpoints/bc_seeds/fs${fs}_seed${sd}.json
  [[ -f "$out" ]] && { echo "  (exists) $(basename $out)"; continue; }
  say "  fs=$fs seed=$sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config /configs/bc_s5d_fs${fs}_seed${sd}.yaml \
      --checkpoint /checkpoints/s5d_fs${fs}_seed${sd}/bc_best.pt \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result' || true
done; done
/scripts/launch_sim_fleet.sh --stop; sleep 4
say "BC SEEDS COMPLETE"
