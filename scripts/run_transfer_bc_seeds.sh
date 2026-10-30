#!/usr/bin/env bash
# The BC half of the transfer test, across training seeds.
#
#   docker exec -d sim /scripts/run_transfer_bc_seeds.sh
#
# run_transfer.sh gave TD3 three seeds per direction and BC only one, which is
# not a fair comparison and cannot support a seed-level test on the paper's
# central claim. These are the two extra seeds per direction; seed 0 is already
# covered there.
set -uo pipefail
LOG=/logs/sweep/transfer.log
STATUS=/logs/sweep/transfer_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/transfer
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

ev () {  # cfg ckpt out
  local cfg=$1 ck=$2 out="$OUT/$3.json"
  [[ -f "$out" ]] && { echo "  (exists) $3"; return 0; }
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result' || true
}

say "waiting for the first transfer queue"
while pgrep -f '[r]un_transfer.sh' > /dev/null; do sleep 90; done

say "5x5-trained BC seeds 1-2, evaluated in the 7x7 world"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 1 2; do
  ev /logs/bc_s4_fs4_seed${sd}.yaml /checkpoints/s4_fs4_seed${sd}/bc_best.pt bc_s4_in_s5_seed${sd}
done
stop_fleet

say "7x7-trained BC seeds 1-2, evaluated in the 5x5 world"
/scripts/launch_sim_fleet.sh -n 1 -s 4 --fast --goal-seed "$GOAL_SEED" || exit 1
for sd in 1 2; do
  ev /configs/bc_s5d_fs4_seed${sd}.yaml /checkpoints/s5d_fs4_seed${sd}/bc_best.pt bc_s5_in_s4_seed${sd}
done
stop_fleet
say "BC SEED TRANSFER COMPLETE"
