#!/usr/bin/env bash
# The complete stage-5 programme, run sequentially with a thermal gate.
#
#   docker exec -d sim /scripts/run_stage5_all.sh
#
# Steps are independent and skip themselves if their output exists, so this is
# safe to re-run after an interruption. Everything here postdates the
# /reset_goal_sequence fix; pre-fix results live in /checkpoints/prefix_*.
#
# THERMAL GATE: before each heavy step, wait for the CPU to drop below GO_TEMP.
# The host writes thermal/thermal.csv once a minute; the container reads the
# same file through the /logs bind mount is NOT available, so the gate reads
# /sys directly, which IS visible to the container.

set -uo pipefail

LOG=/logs/sweep/stage5_all.log
STATUS=/logs/sweep/stage5_status.txt
GO_TEMP=${GO_TEMP:-93}      # wait until at or below this before starting a step
MAX_WAIT=${MAX_WAIT:-1800}  # but never stall the queue more than 30 min
MIN_FREE_GB=4
EPISODES=300
GOAL_SEED=1000

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }

cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }

cool_gate () {
  local waited=0 t
  t=$(cpu_c)
  while (( t > GO_TEMP && waited < MAX_WAIT )); do
    echo "  [thermal] ${t}C > ${GO_TEMP}C - waiting 60s (waited ${waited}s)"
    sleep 60; waited=$((waited + 60)); t=$(cpu_c)
  done
  echo "  [thermal] starting at ${t}C$( (( waited >= MAX_WAIT )) && echo ' (max wait reached)')"
}

room () {
  local free; free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  (( free < MIN_FREE_GB )) && { warn "disk below ${MIN_FREE_GB}G (${free}G)"; return 1; }
  return 0
}

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p /logs/sweep /checkpoints/floor_s5 /checkpoints/substrate_s5_dagger

stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

bc_eval () {  # cfg ckpt action_repeat overrides out
  local cfg=$1 ck=$2 ar=$3 ovr=$4 out=$5
  [[ -f "$out" ]] && { echo "  (exists) $out"; return 0; }
  local flag="--no-overrides"; [[ "$ovr" == "1" ]] && flag=""
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat "$ar" $flag --quiet --out "$out" 2>&1 \
      | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $out"
}

say "STAGE-5 FULL PROGRAMME START"
room || exit 1

# ── 1. DAgger round (real-time pacing; see run_dagger_s5.sh for why) ─────────
say "STEP 1/6  DAgger round on stage 5 (~3.5 h, real-time)"
if [[ ! -f /demos/dagger_s5_iter2.npz ]]; then
  cool_gate
  /scripts/run_dagger_s5.sh 200 || warn "DAgger collection"
else
  echo "  (exists) /demos/dagger_s5_iter2.npz"
fi

# ── 2. merge + retrain BC at both frame stacks ──────────────────────────────
say "STEP 2/6  merge demos and retrain BC (fs=1 and fs=4)"
if [[ -f /demos/dagger_s5_iter2.npz ]]; then
  if [[ ! -f /demos/stage11_dagger_agg2.npz ]]; then
    python3 /bc/dataset_merge.py --inputs /demos/stage11_dagger_agg1.npz /demos/dagger_s5_iter2.npz \
        --out /demos/stage11_dagger_agg2.npz 2>&1 | tail -3 || warn "merge"
  fi
  for fs in 1 4; do
    [[ -f /checkpoints/s5d_fs${fs}/bc_best.pt ]] && { echo "  (exists) s5d_fs${fs}"; continue; }
    sed "s|/checkpoints/s5_fs${fs}|/checkpoints/s5d_fs${fs}|" /configs/bc_s5_fs${fs}.yaml \
        > /configs/bc_s5d_fs${fs}.yaml
    python3 -m bc.train --config /configs/bc_s5d_fs${fs}.yaml \
        --dataset /demos/stage11_dagger_agg2.npz 2>&1 | tail -3 || warn "train fs=$fs"
  done
else
  warn "no DAgger data - skipping retrain, later steps will use the pre-DAgger policy"
fi

# pick whichever policy set exists
PFX=/checkpoints/s5_fs ; CFGP=/configs/bc_s5_fs
[[ -f /checkpoints/s5d_fs4/bc_best.pt ]] && { PFX=/checkpoints/s5d_fs; CFGP=/configs/bc_s5d_fs; }
echo "  evaluating policies from ${PFX}{1,4}"

# ── 3. stage-5 noise floor with that policy ─────────────────────────────────
say "STEP 3/6  stage-5 noise floor (3 x 100, identical config)"
if [[ ! -f /checkpoints/floor_s5/r3.json ]]; then
  cool_gate; room || exit 1
  /scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet"
  for r in 1 2 3; do
    [[ -f /checkpoints/floor_s5/r$r.json ]] && continue
    ROS_DOMAIN_ID=40 timeout 3600 python3 /bc/eval.py --config ${CFGP}4.yaml \
        --checkpoint ${PFX}4/bc_best.pt --episodes 100 --action-repeat 1 \
        --no-overrides --quiet --out /checkpoints/floor_s5/r$r.json 2>&1 \
        | grep -E 'rewound|Result' || true
  done
  stop_fleet
fi

# ── 4. stage-5 BC factorial + wrapper on the DAgger policy ──────────────────
say "STEP 4/6  stage-5 BC factorial + wrapper (6 x ${EPISODES})"
cool_gate; room || exit 1
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet"
for fs in 1 4; do for ar in 1 4; do
  say "  s5d fs=$fs ar=$ar"
  bc_eval ${CFGP}${fs}.yaml ${PFX}${fs}/bc_best.pt "$ar" 0 \
          /checkpoints/substrate_s5_dagger/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s5d wrapper on/off"
bc_eval ${CFGP}4.yaml ${PFX}4/bc_best.pt 1 0 /checkpoints/substrate_s5_dagger/bc_policy_alone.json
bc_eval ${CFGP}4.yaml ${PFX}4/bc_best.pt 1 1 /checkpoints/substrate_s5_dagger/bc_with_overrides.json
stop_fleet

# ── 5. stage-5 TD3 with the SHIPPED config (provenance finding) ─────────────
say "STEP 5/6  stage-5 TD3, shipped config, 3 seeds (~8 h)"
sed -e 's|^  learning_rate: 3.0e-4|  learning_rate: 1.0e-4|' \
    -e 's|^    sigma: 0.1|    sigma: 0.05|' /configs/td3_s5.yaml > /configs/td3_s5_shipped.yaml
for seed in 0 1 2; do
  d=/checkpoints/tune_s5/td3_s5_shipped_seed${seed}
  [[ -f "$d/eval_results.json" ]] && { echo "  seed $seed done"; continue; }
  cool_gate; room || break
  say "  TD3 shipped stage-5 seed $seed"
  /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet seed $seed"; continue; }
  timeout 54000 python3 /rl/sweep_td3.py --config /configs/td3_s5_shipped.yaml --seed "$seed" \
      --n-envs 4 --out-dir "$d" --max-bridge-steps 640000 \
      > "/logs/sweep/td3_s5_shipped_seed${seed}.log" 2>&1 || warn "train seed $seed"
  timeout 10800 python3 /rl/eval_budget.py --run-dir "$d" --config /configs/td3_s5_shipped.yaml \
      --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "eval seed $seed"
  stop_fleet
done

# ── 6. stage-5 AIRL: adversarial phase, then the budget sweep ───────────────
# AIRL is two jobs, not one. Phase 1 (train_airl.py) runs the BC warm start and
# the adversarial loop to produce a reward net; phase 2 (sweep_airl.py)
# optimises a fresh PPO against that frozen reward across the budget grid.
# Stage 5 has no reward net -- the one on disk is stage 4's -- so phase 1 has
# to run first. configs/airl.yaml is already a stage-5 config (6 m lidar).
# frame_stack 1 matches the stage-4 arm, so the trajectories are the fs=1 pickle.
say "STEP 6a/6  stage-5 AIRL adversarial phase (~2 h)"
if [[ ! -f /checkpoints/airl_s5/reward_net.pt ]]; then
  cool_gate; room || exit 1
  /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet (airl p1)"
  timeout 36000 python3 /airl/train_airl.py --config /configs/airl.yaml \
      --trajectories /demos/stage11_trajectories.pkl \
      --out-dir /checkpoints/airl_s5 --frame-stack 1 --seed 0 \
      > /logs/sweep/airl_s5_phase1.log 2>&1 || warn "AIRL phase 1"
  tail -4 /logs/sweep/airl_s5_phase1.log
  stop_fleet
else
  echo "  (exists) /checkpoints/airl_s5/reward_net.pt"
fi

say "STEP 6b/6  stage-5 AIRL budget sweep, 3 seeds (~7 h)"
RNET=/checkpoints/airl_s5/reward_net_best_eval.pt
[[ -f "$RNET" ]] || RNET=/checkpoints/airl_s5/reward_net.pt
IPOL=/checkpoints/airl_s5/policy_after_bc.zip
if [[ -f "$RNET" && -f "$IPOL" ]]; then
  for seed in 0 1 2; do
    d=/checkpoints/sweep_s5/airl_s5_seed${seed}
    [[ -f "$d/eval_results.json" ]] && { echo "  seed $seed done"; continue; }
    cool_gate; room || break
    say "  AIRL stage-5 seed $seed"
    /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet airl $seed"; continue; }
    timeout 43200 python3 /airl/sweep_airl.py --config /configs/airl.yaml \
        --env-config /configs/td3_s5.yaml --reward-net "$RNET" --init-policy "$IPOL" \
        --seed "$seed" --n-envs 4 --frame-stack 1 --out-dir "$d" \
        --max-bridge-steps 640000 \
        > "/logs/sweep/airl_s5_seed${seed}.log" 2>&1 || warn "airl train seed $seed"
    timeout 10800 python3 /rl/eval_budget.py --run-dir "$d" --config /configs/airl.yaml \
        --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "airl eval seed $seed"
    stop_fleet
  done
else
  warn "no stage-5 reward net - AIRL sweep skipped"
fi

say "STAGE-5 FULL PROGRAMME COMPLETE"
