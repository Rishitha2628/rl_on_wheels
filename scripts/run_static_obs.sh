#!/usr/bin/env bash
# Does frame stacking still help when nothing moves?
#
#   docker exec -d sim /scripts/run_static_obs.sh
#
# The paper's stated mechanism for the stacking effect is that a history lets
# the policy infer obstacle velocity. If that is the mechanism, freezing the
# cylinders should remove the benefit. The cylinders are still spawned, at the
# same per-episode sampled positions -- only their motion is switched off
# (dynamic_obstacles:=false), so obstacle count and spatial distribution are
# unchanged and motion is the single varied factor.
#
# Protocol mirrors the moving-obstacle condition exactly (stage 5, goal seed
# 1000, 300 episodes, action_repeat=1) so the cells are directly comparable to
# checkpoints/substrate_s5_dagger (seed 0) and checkpoints/bc_seeds (seeds 1-3).
set -uo pipefail
LOG=/logs/sweep/static_obs.log
STATUS=/logs/sweep/static_obs_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/static_obs
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

say "STATIC-OBSTACLE TEST  (stacking x obstacle motion, 4 seeds)"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" --static-obs || exit 1
# Confirm the obstacle node really is absent before spending two hours.
sleep 3
if ROS_DOMAIN_ID=40 ros2 node list 2>/dev/null | grep -q dynamic_obstacle; then
  echo "ABORT: dynamic_obstacle_node is running -- obstacles are NOT static." >&2
  /scripts/launch_sim_fleet.sh --stop; exit 1
fi
say "confirmed: no dynamic_obstacle_node (cylinders frozen)"

for fs in 1 4; do for sd in 0 1 2 3; do
  out="$OUT/fs${fs}_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) fs=$fs seed=$sd"; continue; }
  if   [[ $sd == 0 ]];                                    then cfg=/configs/bc_s5d_fs${fs}.yaml
  elif [[ -f /configs/bc_s5d_fs${fs}_seed${sd}.yaml ]];   then cfg=/configs/bc_s5d_fs${fs}_seed${sd}.yaml
  else                                                         cfg=/logs/bc_s5d_fs${fs}_seed${sd}.yaml; fi
  ck=/checkpoints/s5d_fs${fs}$([[ $sd == 0 ]] || echo _seed${sd})/bc_best.pt
  [[ -f "$cfg" && -f "$ck" ]] || { echo "  SKIP fs=$fs seed=$sd (cfg=$cfg ck=$ck)"; continue; }
  say "  fs=$fs seed=$sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result' || true
done; done

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "STATIC-OBSTACLE TEST COMPLETE"
