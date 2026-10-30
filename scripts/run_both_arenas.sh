#!/usr/bin/env bash
# Does training on two floor plans buy generalisation to a third?
#
#   docker exec -d sim /scripts/run_both_arenas.sh
#
# Waits for the blank-LiDAR diagnostic to clear -- one sim instance.
#
# The 5x5 and 7x7 demonstration sets are combined by bc/merge_arenas.py, which
# rebases the 5x5 data from its 3.5 m normaliser onto the 6.0 m one so that the
# same number means the same distance in both halves. Both halves therefore
# describe a 3.5 m sensor, and evaluation applies --lidar-max-range 3.5 to
# match.
#
# The comparison is against the SINGLE-arena policy trained on the same 3.5 m
# representation (checkpoints/range_transfer): home 53.6%, unseen 19.6%.
# Range is held constant, so the only thing that changes is how many floor
# plans the policy saw.
set -uo pipefail
LOG=/logs/sweep/both_arenas.log
STATUS=/logs/sweep/both_arenas_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/both_arenas
DATA=/demos/both_arenas_r35.npz
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep
[[ -f "$DATA" ]] || { echo "missing $DATA -- run bc/merge_arenas.py" >&2; exit 1; }

for _ in $(seq 1 480); do
  pgrep -f 'run_blank_lidar' >/dev/null 2>&1 || break
  sleep 30
done
pgrep -f 'run_blank_lidar' >/dev/null 2>&1 && { echo "ABORT: blank-lidar still running" >&2; exit 1; }
say "blank-lidar clear; starting two-arena training"

for sd in 0 1 2; do
  ck=/checkpoints/both_arenas_seed${sd}/bc_best.pt
  [[ -f "$ck" ]] && { echo "  (exists) train seed $sd"; continue; }
  say "  training seed $sd on both arenas"
  python3 -m bc.train --config /configs/bc_both_arenas_seed${sd}.yaml \
      --dataset "$DATA" --seed "$sd" 2>&1 | tail -2
done

/scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4
for stage in 5 6; do
  tag=$([[ $stage == 5 ]] && echo home7x7 || echo unseen)
  say "stage $stage ($tag)"
  /scripts/launch_sim_fleet.sh -n 1 -s "$stage" --fast --goal-seed "$GOAL_SEED" || exit 1
  for sd in 0 1 2; do
    out="$OUT/both_${tag}_seed${sd}.json"
    [[ -f "$out" ]] && { echo "  (exists) $tag seed $sd"; continue; }
    say "  $tag seed $sd"
    ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
        --config /configs/bc_both_arenas_seed${sd}.yaml \
        --checkpoint /checkpoints/both_arenas_seed${sd}/bc_best.pt \
        --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
        --lidar-max-range 3.5 --out "$out" 2>&1 | grep -E 'substrate|rewound|Result' || true
  done
  /scripts/launch_sim_fleet.sh --stop; sleep 4
done
say "TWO-ARENA TRANSFER COMPLETE"
