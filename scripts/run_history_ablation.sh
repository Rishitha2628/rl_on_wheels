#!/usr/bin/env bash
# Which PART of the frame stack carries the stacking benefit?
#
#   docker exec -d sim /scripts/run_history_ablation.sh
#
# Four stacked frames beat one by 5.5 points over four seeds each (51.6 -> 57.1).
# The 40-dim BC observation splits into two groups that could explain it:
#
#   scan  [0..35] lidar bins + [39] the goal-cone range
#         history here shows how obstacles move relative to the robot
#   self  [36] goal distance, [37..38] goal bearing
#         history here IS the robot's own motion, differenced over frames
#
# There is no action group. build_dataset.py omits prev_lin_vel/prev_av on
# purpose and bc/eval.py slices the live 42-dim observation to 40 for the same
# reason, so this policy has never seen an action, its own or the expert's. The
# self group is the nearest thing the observation holds to "what have I been
# doing".
#
# Holding a group's history out keeps the input at 160 dims and copies that
# group's current-frame values into the three past frames. Capacity is
# therefore identical across the three cells and to the shipped 4-frame policy,
# so a difference is about information and not model size. bc/eval.py reads the
# setting from the checkpoint, and bc/verify_history_mask.py proves the offline
# and live paths build identical inputs.
#
#   none  no history at all (a capacity control for the 1-frame cell)
#   scan  lidar history only
#   self  self-motion history only
#
# Read against the two cells we already have, both over four seeds:
#   1 frame  51.6%   4 full frames  57.1%
#   scan ~ full and self ~ none  -> perception of obstacle motion
#   self ~ full and scan ~ none  -> knowing its own motion
#   both partial                 -> both contribute
#
# THREE SEEDS PER CELL, FIXED BEFORE THE RUNS, REPORTING WHATEVER COMES OUT.
set -uo pipefail
LOG=/logs/sweep/history_ablation.log
STATUS=/logs/sweep/history_ablation_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; EPISODES=300; GOAL_SEED=1000
DATASET=/demos/stage11_dagger_agg2.npz
OUT=/checkpoints/hist_ablation

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

[[ -f "$DATASET" ]] || { warn "missing $DATASET"; exit 1; }

say "checking the offline and live masking paths agree"
python3 /bc/verify_history_mask.py || { warn "mask verification"; exit 1; }

say "training 3 history settings x 3 seeds"
for cell in none scan self; do for sd in 0 1 2; do
  d=/checkpoints/hist_${cell}_seed${sd}
  [[ -f "$d/bc_best.pt" ]] && { echo "  (exists) $cell seed=$sd"; continue; }
  echo "  training $cell seed=$sd"
  python3 -m bc.train --config "/configs/bc_hist_${cell}_seed${sd}.yaml" \
      --dataset "$DATASET" --seed "$sd" --history "$cell" 2>&1 | tail -1
  [[ -f "$d/bc_best.pt" ]] || { warn "training $cell seed=$sd"; exit 1; }
done; done

cool_gate
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet"; exit 1; }

say "evaluating on the same 300 episodes as every other 7x7 cell"
for cell in none scan self; do for sd in 0 1 2; do
  out="$OUT/${cell}_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) $(basename "$out")"; continue; }
  say "  $cell seed=$sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config "/configs/bc_hist_${cell}_seed${sd}.yaml" \
      --checkpoint "/checkpoints/hist_${cell}_seed${sd}/bc_best.pt" \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result|history|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $cell seed=$sd"
done; done

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "HISTORY ABLATION COMPLETE"
python3 /scripts/history_tests.py || true
