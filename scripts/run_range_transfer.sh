#!/usr/bin/env bash
# Does a shorter LiDAR range make the policy generalise?
#
#   docker exec -d sim /scripts/run_range_transfer.sh
#
# The observation normalises LiDAR by 6.0 m, so in the 7x7 arena the policy can
# see most of the room and may be fitting the layout rather than the task. The
# real TurtleBot3 LDS-01 reaches 3.5 m. We clip the recorded demonstrations to
# 3.5 m (66% of bins are beyond it), retrain, and evaluate at home and in the
# unseen arena with the same clipping applied at evaluation time.
#
# The clipping is done in the DATA, not by lowering max_lidar_range, because
# that constant also normalises the goal distance (env_bridge_node.cpp) and
# would confound a range change with a goal-encoding change. bc/clip_lidar_range.py
# reproduces LidarDegradation's transform exactly (verified bit-identical), so
# training and evaluation see the same distribution.
#
# Baseline to beat (unclipped, same seeds): 61.3% at home, 17.0% unseen, -44.3.
set -uo pipefail
LOG=/logs/sweep/range_transfer.log
STATUS=/logs/sweep/range_transfer_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/range_transfer
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

DATA=/demos/stage11_dagger_agg2_r35.npz
[[ -f "$DATA" ]] || { echo "missing $DATA -- run bc/clip_lidar_range.py first" >&2; exit 1; }

say "RANGE-CLIPPED TRANSFER TEST (3.5 m)"

# ── train three seeds on the clipped demonstrations ─────────────────────────
for sd in 0 1 2; do
  ck=/checkpoints/s5d_fs4_r35_seed${sd}/bc_best.pt
  [[ -f "$ck" ]] && { echo "  (exists) train seed $sd"; continue; }
  say "  training seed $sd on clipped demos"
  python3 -m bc.train --config /configs/bc_s5d_fs4_r35_seed${sd}.yaml \
      --dataset "$DATA" --seed "$sd" 2>&1 | tail -2
done

# ── evaluate at home (7x7) ──────────────────────────────────────────────────
say "evaluating at home (7x7), range clipped at eval too"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 0 1 2; do
  out="$OUT/r35_home_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) home seed $sd"; continue; }
  say "  home seed $sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config /configs/bc_s5d_fs4_r35_seed${sd}.yaml \
      --checkpoint /checkpoints/s5d_fs4_r35_seed${sd}/bc_best.pt \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --lidar-max-range 3.5 --out "$out" 2>&1 | grep -E 'substrate|rewound|Result' || true
done
/scripts/launch_sim_fleet.sh --stop; sleep 4

# ── evaluate in the unseen arena (stage 6) ──────────────────────────────────
say "evaluating in the unseen arena (stage 6)"
/scripts/launch_sim_fleet.sh -n 1 -s 6 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 0 1 2; do
  out="$OUT/r35_unseen_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) unseen seed $sd"; continue; }
  say "  unseen seed $sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config /configs/bc_s5d_fs4_r35_seed${sd}.yaml \
      --checkpoint /checkpoints/s5d_fs4_r35_seed${sd}/bc_best.pt \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --lidar-max-range 3.5 --out "$out" 2>&1 | grep -E 'substrate|rewound|Result' || true
done
/scripts/launch_sim_fleet.sh --stop; sleep 4
say "RANGE-CLIPPED TRANSFER COMPLETE"
