#!/usr/bin/env bash
# Stage-5 AIRL, both phases. Runs after the fixup queue so they never share a fleet.
#
#   docker exec -d sim /scripts/run_airl_s5.sh
#
# Why the first attempt failed: train_airl.py builds a DummyVecEnv with ONE env
# and talks to whatever ROS_DOMAIN_ID the shell has. The queue launched a
# 4-instance fleet on domains 40-43 but never set ROS_DOMAIN_ID, so the trainer
# looked for /get_observation on domain 0 and found nothing. Phase 1 therefore
# needs exactly one sim, on domain 40, with ROS_DOMAIN_ID exported.
# Phase 2 (sweep_airl.py) does take --n-envs/--domain-base and uses the fleet.
set -uo pipefail
LOG=/logs/sweep/airl_s5.log
STATUS=/logs/sweep/airl_s5_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; GOAL_SEED=1000

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:/airl:${PYTHONPATH:-}
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

say "waiting for the fixup queue"
while pgrep -f '[r]un_stage5_fixup.sh' > /dev/null; do sleep 120; done
say "fixup done - starting AIRL"

# ── phase 1: adversarial, ONE sim, domain 40 ────────────────────────────────
say "PHASE 1  adversarial (1 sim, ROS_DOMAIN_ID=40)"
if [[ ! -f /checkpoints/airl_s5/reward_net.pt ]]; then
  cool_gate
  /scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet p1"
  ROS_DOMAIN_ID=40 timeout 36000 python3 /airl/train_airl.py --config /configs/airl.yaml \
      --trajectories /demos/stage11_trajectories.pkl \
      --out-dir /checkpoints/airl_s5 --frame-stack 1 --seed 0 \
      > /logs/sweep/airl_s5_phase1.log 2>&1 || warn "AIRL phase 1"
  tail -5 /logs/sweep/airl_s5_phase1.log
  stop_fleet
else
  echo "  (exists) reward_net.pt"
fi

# ── phase 2: budget sweep against the frozen reward ─────────────────────────
RNET=/checkpoints/airl_s5/reward_net_best_eval.pt
[[ -f "$RNET" ]] || RNET=/checkpoints/airl_s5/reward_net.pt
IPOL=/checkpoints/airl_s5/policy_after_bc.zip
if [[ ! -f "$RNET" ]]; then
  warn "no stage-5 reward net - phase 2 skipped"
  say "AIRL INCOMPLETE"; exit 1
fi

say "PHASE 2  budget sweep, 3 seeds"
for seed in 0 1 2; do
  d=/checkpoints/sweep_s5/airl_s5_seed${seed}
  [[ -f "$d/eval_results.json" ]] && { echo "  seed $seed done"; continue; }
  cool_gate
  say "  AIRL seed $seed"
  /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet seed $seed"; continue; }
  timeout 43200 python3 /airl/sweep_airl.py --config /configs/airl.yaml \
      --env-config /configs/td3_s5.yaml --reward-net "$RNET" --init-policy "$IPOL" \
      --seed "$seed" --n-envs 4 --domain-base 40 --frame-stack 1 --out-dir "$d" \
      --max-bridge-steps 640000 > "/logs/sweep/airl_s5_seed${seed}.log" 2>&1 || warn "train seed $seed"
  timeout 10800 python3 /rl/eval_budget.py --run-dir "$d" --config /configs/airl.yaml \
      --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "eval seed $seed"
  stop_fleet
done
say "AIRL COMPLETE"
