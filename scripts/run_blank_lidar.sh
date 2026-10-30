#!/usr/bin/env bash
# Does the policy actually use its LiDAR, or steer on the goal features alone?
#
#   docker exec -d sim /scripts/run_blank_lidar.sh
#
# Every LiDAR bin is forced to max range ("no obstacle anywhere") at evaluation
# time, in the home arena and the unseen one, with everything else unchanged.
# If success barely moves, the policy is navigating on goal distance and
# bearing and the LiDAR is close to decorative -- which would support the
# memorisation reading of the transfer collapse. If it collapses, the policy
# genuinely depends on the scan.
#
# --lidar-dropout 1.0 sets every bin to max range, which is how the bridge
# represents a no-return; no new code is needed.
#
# Reference (same seeds, undegraded): home 61.3%, unseen 17.0%.
set -uo pipefail
LOG=/logs/sweep/blank_lidar.log
STATUS=/logs/sweep/blank_lidar_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/blank_lidar
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

cfg_for () { [[ $1 == 0 ]] && echo /configs/bc_s5d_fs4.yaml || echo /configs/bc_s5d_fs4_seed$1.yaml; }
ck_for ()  { [[ $1 == 0 ]] && echo /checkpoints/s5d_fs4/bc_best.pt || echo /checkpoints/s5d_fs4_seed$1/bc_best.pt; }

say "BLANK-LIDAR DIAGNOSTIC"
for stage in 5 6; do
  tag=$([[ $stage == 5 ]] && echo home || echo unseen)
  say "stage $stage ($tag)"
  /scripts/launch_sim_fleet.sh -n 1 -s "$stage" --fast --goal-seed "$GOAL_SEED" || exit 1
  for sd in 0 1 2; do
    out="$OUT/blank_${tag}_seed${sd}.json"
    [[ -f "$out" ]] && { echo "  (exists) $tag seed $sd"; continue; }
    say "  $tag seed $sd"
    ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
        --config "$(cfg_for $sd)" --checkpoint "$(ck_for $sd)" \
        --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
        --lidar-dropout 1.0 --out "$out" 2>&1 | grep -E 'substrate|rewound|Result' || true
  done
  /scripts/launch_sim_fleet.sh --stop; sleep 4
done
say "BLANK-LIDAR COMPLETE"
