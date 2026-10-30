#!/usr/bin/env bash
# Compare the reward AIRL recovered against the one we hand-wrote.
# Offline -- no simulator -- so it waits only to keep the logs tidy.
#
#   docker exec -d sim /scripts/run_reward_compare.sh
set -uo pipefail
exec > >(stdbuf -oL tee -a /logs/sweep/reward_compare.log) 2>&1
set +u; source /opt/ros/humble/setup.bash; set -u
export PYTHONPATH=/:${PYTHONPATH:-}
echo "=== $(date '+%a %H:%M:%S')  waiting for the transfer queues"
while pgrep -f '[r]un_transfer' > /dev/null; do sleep 120; done
echo "=== $(date '+%a %H:%M:%S')  comparing rewards"
python3 /airl/compare_rewards.py \
    --checkpoint-dir /checkpoints/airl_s5 \
    --demos /demos/stage11_dagger_agg2.npz \
    --out /checkpoints/reward_cmp 2>&1 | tail -5
echo "=== $(date '+%a %H:%M:%S')  REWARD COMPARISON DONE"
