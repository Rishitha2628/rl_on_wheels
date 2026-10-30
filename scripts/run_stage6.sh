#!/usr/bin/env bash
# The behaviour test: policies trained in stage 5 (7x7), evaluated in stage 6 --
# a world they have never seen.
#
#   docker exec -d sim /scripts/run_stage6.sh
#
# Stage 6 is not a variant of stage 5. It is a 6x6 room split into chambers by
# six 2 m dividers with four moving cylinders, against stage 5's 7x7 box with
# eight 1 m stubs and six cylinders: different size, different obstacle
# geometry, different count, different topology. A policy that learned "reach
# the goal, avoid things" should cope; one that learned stage 5's floor plan
# should not.
#
# Each policy keeps its own config, so its observation scaling is the one it
# trained with and only the world changes.
set -uo pipefail
LOG=/logs/sweep/stage6.log
STATUS=/logs/sweep/stage6_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/stage6
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

say "STAGE 6 BEHAVIOUR TEST"

# ── TD3 first: it uses the 4-instance fleet and is ~4x faster per policy ────
say "TD3 (stage-5 trained), 3 seeds, in stage 6"
/scripts/launch_sim_fleet.sh -n 4 -s 6 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 0 1 2; do
  out="$OUT/td3_s5_in_s6_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) seed $sd"; continue; }
  timeout 7200 python3 /rl/eval_transfer.py \
      --checkpoint /checkpoints/tune_s5/td3_s5_seed${sd}/budget_640000.zip \
      --config /configs/td3_s5.yaml --episodes "$EPISODES" --n-envs 4 \
      --out "$out" --note "TD3 7x7->stage6 seed ${sd}" 2>&1 | tail -2
done
stop_fleet

# ── BC: single-env, so one sim is enough and four would idle ────────────────
say "BC+DAgger (stage-5 trained), 3 seeds, in stage 6"
/scripts/launch_sim_fleet.sh -n 1 -s 6 --fast --goal-seed "$GOAL_SEED" || exit 1
for spec in "s5d_fs4:bc_s5d_fs4:0" "s5d_fs4_seed1:bc_s5d_fs4_seed1:1" "s5d_fs4_seed2:bc_s5d_fs4_seed2:2"; do
  ck=${spec%%:*}; rest=${spec#*:}; cfg=${rest%%:*}; sd=${rest#*:}
  out="$OUT/bc_s5_in_s6_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) seed $sd"; continue; }
  say "  BC seed $sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config /configs/${cfg}.yaml --checkpoint /checkpoints/${ck}/bc_best.pt \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result' || true
done
stop_fleet
say "STAGE 6 COMPLETE"
