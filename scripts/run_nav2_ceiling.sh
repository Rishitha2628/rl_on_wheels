#!/usr/bin/env bash
# What can the teacher do in the unseen arena?
#
#   docker exec -d sim /scripts/run_nav2_ceiling.sh
#
# Nav2 drives (beta=1.0, so the learned policy's output is never executed) with
# the stage-6 occupancy map, in the same arena, goal sequence and success
# criterion as every other evaluation. It separates two readings of the
# transfer failure that our policy results cannot:
#
#   Nav2 succeeds  -> the task is solvable there and the arena is not simply
#                     too hard; the policies failed to learn it.
#   Nav2 struggles -> the arena is hard for a planner too, and the gap is
#                     partly the task rather than the learning.
#
# Nav2 has a map and a global planner, so this is a privileged ceiling, not a
# like-for-like baseline. The hand-written safety stack is active, matching the
# conditions under which the demonstrations were collected.
#
# Reference: BC+DAgger 17.0% and TD3 19.9% in this arena; a straight-line
# robot 14.4%.
set -uo pipefail
LOG=/logs/sweep/nav2_ceiling.log
GOAL_SEED=1000; EPISODES=150; OUT=/checkpoints/nav2_ceiling
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

for stage in 6 5; do
  tag=$([[ $stage == 6 ]] && echo unseen || echo home)
  out="$OUT/nav2_${tag}.npz"
  [[ -f "${out%.npz}_done" ]] && { echo "  (exists) $tag"; continue; }

  say "Nav2 in stage $stage ($tag)"
  /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4
  /scripts/launch_sim_fleet.sh -n 1 -s "$stage" --rt --goal-seed "$GOAL_SEED" || exit 1

  say "  bringing up Nav2 (stage $stage)"
  # IGN_PARTITION matters as much as ROS_DOMAIN_ID: pose_publisher_node reads
  # the robot's ground-truth pose over gz-transport, not ROS, so without the
  # partition it subscribes to an empty topic and never publishes map->odom.
  ROS_DOMAIN_ID=40 IGN_PARTITION=tb3env0 setsid ros2 launch tb3_nav2 nav2_bringup.launch.py \
      stage:="$stage" dagger_mode:=true > /logs/sweep/nav2_bringup_${stage}.log 2>&1 &
  NAV=$!
  sleep 45

  say "  running $EPISODES episodes, Nav2 driving (beta=1.0)"
  ROS_DOMAIN_ID=40 IGN_PARTITION=tb3env0 timeout 10800 python3 /bc/dagger_collect.py \
      --config /configs/bc_s5d_fs4.yaml \
      --checkpoint /checkpoints/s5d_fs4/bc_best.pt \
      --episodes "$EPISODES" --beta 1.0 --out "$out" 2>&1 \
      | grep -E 'success-rate|\[ep .*(50|100|150)/' || true
  touch "${out%.npz}_done"
  kill -TERM -$NAV 2>/dev/null || true
  sleep 5
done
/scripts/launch_sim_fleet.sh --stop; sleep 3
say "NAV2 CEILING COMPLETE"
