#!/usr/bin/env bash
# Did the policies learn the behaviour, or memorise the world?
#
#   docker exec -d sim /scripts/run_transfer.sh
#
# Every evaluation in this project so far scored a policy in the world it was
# trained in, so "TD3 memorised the arena, BC+DAgger learned to navigate" was
# an assertion. Here each policy is evaluated in BOTH arenas. The policy always
# keeps its own config, so its observation scaling (max_lidar_range 3.5 on the
# 5x5, 6.0 on the 7x7) is the one it learned with; only the geometry changes.
#
#   BC  5x5-trained -> 7x7      TD3 5x5-trained -> 7x7   (3 seeds)
#   BC  7x7-trained -> 5x5      TD3 7x7-trained -> 5x5   (3 seeds)
#
# Own-world numbers already exist, so the comparison is drop-from-home.
set -uo pipefail
LOG=/logs/sweep/transfer.log
STATUS=/logs/sweep/transfer_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/transfer
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

bc_in_world () {   # world_stage config ckpt out note
  local st=$1 cfg=$2 ck=$3 out="$OUT/$4.json" note=$5
  [[ -f "$out" ]] && { echo "  (exists) $4"; return 0; }
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result' || true
}
td3_in_world () {  # config ckpt out note
  local cfg=$1 ck=$2 out="$OUT/$3.json" note=$4
  [[ -f "$out" ]] && { echo "  (exists) $3"; return 0; }
  timeout 7200 python3 /rl/eval_transfer.py --checkpoint "$ck" --config "$cfg" \
      --episodes "$EPISODES" --n-envs 4 --out "$out" --note "$note" 2>&1 | tail -2
}

# ── 5x5-trained policies, evaluated in the 7x7 world ────────────────────────
say "5x5-trained policies in the 7x7 world"
/scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1
bc_in_world 5 /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt bc_s4_in_s5 "BC 5x5->7x7"
for sd in 0 1 2; do
  td3_in_world /configs/td3_tune_a.yaml \
      /checkpoints/tune/td3_confirm_seed${sd}/budget_640000.zip \
      td3_s4_in_s5_seed${sd} "TD3 5x5->7x7 seed ${sd}"
done
stop_fleet

# ── 7x7-trained policies, evaluated in the 5x5 world ────────────────────────
say "7x7-trained policies in the 5x5 world"
/scripts/launch_sim_fleet.sh -n 4 -s 4 --fast --goal-seed "$GOAL_SEED" || exit 1
bc_in_world 4 /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt bc_s5_in_s4 "BC 7x7->5x5"
for sd in 0 1 2; do
  td3_in_world /configs/td3_s5.yaml \
      /checkpoints/tune_s5/td3_s5_seed${sd}/budget_640000.zip \
      td3_s5_in_s4_seed${sd} "TD3 7x7->5x5 seed ${sd}"
done
stop_fleet
say "TRANSFER COMPLETE"
