#!/usr/bin/env bash
# The seed-level tests the paper still owes, plus one pre-committed addition.
#
#   docker exec -d sim /scripts/run_seed_closeout.sh
#
# SEED COUNTS ARE FIXED HERE, BEFORE THE RUNS, AND WE REPORT WHATEVER COMES OUT:
#   stacking (7x7, 1 vs 4 frames)      : 4 seeds   (3 exist, +1 here)
#   wrapper  (7x7, with vs without)    : 3 seeds   (1 exists, +2 here)
#   wrapper  (5x5, with vs without)    : 3 seeds   (1 exists, +2 here)
# Adding seeds until a result turns significant would be the same selection
# effect the paper is about, so the counts are committed in advance.
set -uo pipefail
LOG=/logs/sweep/seed_closeout.log
STATUS=/logs/sweep/seed_closeout_status.txt
GOAL_SEED=1000; EPISODES=300
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p /checkpoints/bc_seeds /checkpoints/wrap_seeds /logs/sweep

ev () {  # cfg ckpt overrides out
  local cfg=$1 ck=$2 ovr=$3 out=$4
  [[ -f "$out" ]] && { echo "  (exists) $(basename $out)"; return 0; }
  local flag="--no-overrides"; [[ "$ovr" == "1" ]] && flag=""
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat 1 $flag --quiet --out "$out" 2>&1 \
      | grep -E 'rewound|Result' || true
}

say "training the policies these tests need"
# 4th stacking seed on 7x7
for fs in 1 4; do
  d=/checkpoints/s5d_fs${fs}_seed3
  [[ -f "$d/bc_best.pt" ]] || {
    sed "s|/checkpoints/s5d_fs${fs}|/checkpoints/s5d_fs${fs}_seed3|" /configs/bc_s5d_fs${fs}.yaml > /tmp/bc_s5d_fs${fs}_seed3.yaml
    cp /tmp/bc_s5d_fs${fs}_seed3.yaml /logs/bc_s5d_fs${fs}_seed3.yaml
    python3 -m bc.train --config /logs/bc_s5d_fs${fs}_seed3.yaml \
        --dataset /demos/stage11_dagger_agg2.npz --seed 3 2>&1 | tail -1; }
done
# two more 5x5 seeds for the wrapper test
for sd in 1 2; do
  d=/checkpoints/s4_fs4_seed${sd}
  [[ -f "$d/bc_best.pt" ]] || {
    sed "s|checkpoint_dir:   \"/checkpoints\"|checkpoint_dir:   \"/checkpoints/s4_fs4_seed${sd}\"|" /configs/bc_fs4.yaml > /logs/bc_s4_fs4_seed${sd}.yaml
    python3 -m bc.train --config /logs/bc_s4_fs4_seed${sd}.yaml \
        --dataset /demos/stage4_dagger_agg2.npz --seed "$sd" 2>&1 | tail -1; }
done

say "7x7 evaluations"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
say "  stacking seed 3"
ev /logs/bc_s5d_fs1_seed3.yaml /checkpoints/s5d_fs1_seed3/bc_best.pt 0 /checkpoints/bc_seeds/fs1_seed3.json
ev /logs/bc_s5d_fs4_seed3.yaml /checkpoints/s5d_fs4_seed3/bc_best.pt 0 /checkpoints/bc_seeds/fs4_seed3.json
say "  wrapper, seeds 1-2 (seed 0 already measured)"
for sd in 1 2; do
  ev /configs/bc_s5d_fs4_seed${sd}.yaml /checkpoints/s5d_fs4_seed${sd}/bc_best.pt 1 \
     /checkpoints/wrap_seeds/s5_wrapped_seed${sd}.json
done
/scripts/launch_sim_fleet.sh --stop; sleep 4

say "5x5 evaluations"
/scripts/launch_sim_fleet.sh -n 1 -s 4 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 1 2; do
  ev /logs/bc_s4_fs4_seed${sd}.yaml /checkpoints/s4_fs4_seed${sd}/bc_best.pt 0 \
     /checkpoints/wrap_seeds/s4_alone_seed${sd}.json
  ev /logs/bc_s4_fs4_seed${sd}.yaml /checkpoints/s4_fs4_seed${sd}/bc_best.pt 1 \
     /checkpoints/wrap_seeds/s4_wrapped_seed${sd}.json
done
/scripts/launch_sim_fleet.sh --stop; sleep 4
say "SEED CLOSEOUT COMPLETE"
