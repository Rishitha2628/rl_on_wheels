#!/usr/bin/env bash
# Regenerate the AIRL prerequisites that the budget sweep needs.
#
# The original reward_net.pt and policy_after_bc.zip no longer exist: nothing
# under checkpoints/ was ever committed, because .gitignore excludes *.zip.
# What survived is what actually cost something to produce -- the Nav2
# demonstrations in demos/ -- so both artifacts can be rebuilt from those
# without re-running hundreds of Nav2 episodes.
#
# Two stages:
#   1. convert demos   NPZ -> imitation Trajectory pickle. Offline, seconds.
#   2. train_airl.py   BC warm-start + adversarial loop -> reward_net.pt
#                      and policy_after_bc.zip. NEEDS THE SIM.
#
# Stage 2 competes with anything else using the fleet, so do not run this
# while the TD3 sweep is going.
#
# Usage:
#   scripts/prepare_airl.sh                       # stage 4, dagger aggregate
#   scripts/prepare_airl.sh --npz /demos/stage4_bc.npz --convert-only

set -uo pipefail

NPZ=/demos/stage4_dagger_agg2.npz
TRAJ=/demos/stage4_trajectories.pkl
OUT_DIR=/checkpoints/airl_stage4
FRAME_STACK=1
AIRL_STEPS=60000
BC_EPOCHS=20
EVAL_EVERY=5
SEED=0
N_ENVS=4
STAGE=4
CONVERT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npz)          NPZ="$2"; shift 2 ;;
    --traj)         TRAJ="$2"; shift 2 ;;
    --out-dir)      OUT_DIR="$2"; shift 2 ;;
    --frame-stack)  FRAME_STACK="$2"; shift 2 ;;
    --airl-steps)   AIRL_STEPS="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --stage)        STAGE="$2"; shift 2 ;;
    --convert-only) CONVERT_ONLY=1; shift ;;
    --smoke)        SMOKE=1; shift ;;
    -h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Smoke mode: exercise the whole code path cheaply to find out whether
# train_airl.py still runs at all. It has not been executed in months and is
# the only untested link in the pipeline; discovering it is broken now is worth
# far more than the few minutes it costs.
if [[ -n "${SMOKE:-}" ]]; then
  AIRL_STEPS=2000
  BC_EPOCHS=2
  OUT_DIR=/checkpoints/airl_smoke
  echo "[prepare-airl] SMOKE MODE — ${AIRL_STEPS} steps, ${BC_EPOCHS} BC epochs -> $OUT_DIR"
  echo "[prepare-airl] This validates the code path only; the artifacts it"
  echo "[prepare-airl] produces are NOT usable for the sweep."
fi

# Isolated sim instance, well clear of the sweep's fleet on domains 40-43.
EVAL_DOMAIN=51
EVAL_PARTITION=tb3airl

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

if [[ ! -f "$NPZ" ]]; then
  echo "ERROR demo set not found: $NPZ" >&2
  echo "Available:" >&2; ls /demos/*.npz >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# ── stage 1: demos -> trajectories ────────────────────────────────────────────
echo "[prepare-airl] converting $NPZ -> $TRAJ (frame_stack=$FRAME_STACK)"
python3 /airl/convert_demos.py \
    --npz "$NPZ" --out "$TRAJ" --frame-stack "$FRAME_STACK" || {
  echo "[prepare-airl] demo conversion failed" >&2; exit 1; }

if [[ $CONVERT_ONLY -eq 1 ]]; then
  echo "[prepare-airl] --convert-only set; stopping before the sim stage."
  exit 0
fi

# ── stage 2: BC warm-start + adversarial AIRL ─────────────────────────────────
# train_airl.py is single-env, so it gets its own isolated instance rather than
# a fleet. Launched inline, NOT through launch_sim_fleet.sh: that script begins
# by tearing down every sim on the machine (a global `pkill -9 -f "ign
# gazebo"`), which would kill a running sweep. Nothing below touches a process
# it did not start.
if pgrep -f sweep_td3.py >/dev/null 2>&1; then
  echo "[prepare-airl] A sweep is running; sharing the machine."
  echo "[prepare-airl]   Own sim on ROS_DOMAIN_ID=$EVAL_DOMAIN, isolated from"
  echo "[prepare-airl]   the fleet on 40-43. Costs the sweep some throughput;"
  echo "[prepare-airl]   changes none of its results, since robot-hours come"
  echo "[prepare-airl]   from step counts rather than wall clock."
  echo ""
fi

cleanup_airl_sim () {
  pkill -TERM -P $$ 2>/dev/null
  if [[ -n "${AIRL_PGID:-}" ]]; then
    kill -TERM "-$AIRL_PGID" 2>/dev/null
    sleep 3
    kill -9 "-$AIRL_PGID" 2>/dev/null
  fi
  pkill -9 -P $$ 2>/dev/null
}
trap cleanup_airl_sim EXIT INT TERM

echo "[prepare-airl] starting isolated sim (domain $EVAL_DOMAIN, partition $EVAL_PARTITION)"
setsid env \
  ROS_DOMAIN_ID="$EVAL_DOMAIN" \
  IGN_PARTITION="$EVAL_PARTITION" \
  ros2 launch tb3_rl_bridge bridge.launch.py \
    stage:="$STAGE" headless:=true fast:=true \
  > /logs/sweep/prepare_airl_sim.log 2>&1 &
AIRL_PGID=$!

echo "[prepare-airl] waiting for /step on domain $EVAL_DOMAIN ..."
ready=0
for _ in $(seq 1 40); do
  if ROS_DOMAIN_ID="$EVAL_DOMAIN" timeout 8 ros2 service list 2>/dev/null \
       | grep -q '^/step$'; then ready=1; break; fi
  sleep 3
done
if [[ $ready -eq 0 ]]; then
  echo "[prepare-airl] sim did not come up; see /logs/sweep/prepare_airl_sim.log" >&2
  exit 1
fi
echo "[prepare-airl] sim ready."

echo "[prepare-airl] train_airl.py: ${AIRL_STEPS} steps, ${BC_EPOCHS} BC epochs"
ROS_DOMAIN_ID="$EVAL_DOMAIN" IGN_PARTITION="$EVAL_PARTITION" \
  stdbuf -oL -eL python3 /airl/train_airl.py \
    --config /configs/airl.yaml \
    --trajectories "$TRAJ" \
    --total-timesteps "$AIRL_STEPS" \
    --bc-epochs "$BC_EPOCHS" \
    --eval-every "$EVAL_EVERY" \
    --frame-stack "$FRAME_STACK" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" 2>&1 | tee /logs/sweep/prepare_airl.log

status=${PIPESTATUS[0]}
cleanup_airl_sim
trap - EXIT INT TERM

if [[ $status -ne 0 ]]; then
  echo "[prepare-airl] train_airl.py failed (status $status)" >&2
  exit $status
fi

echo ""
echo "[prepare-airl] artifacts:"
for f in "$OUT_DIR/reward_net.pt" "$OUT_DIR/reward_net_best_eval.pt" \
         "$OUT_DIR/policy_after_bc.zip" "$OUT_DIR/policy_best_eval.zip"; do
  if [[ -f "$f" ]]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done

# Phase 2 should freeze the reward net that was paired with the BEST generator,
# not the one left at the end of a loop that may have degraded. If the
# best-eval snapshot exists, promote it over the final one.
if [[ -f "$OUT_DIR/reward_net_best_eval.pt" ]]; then
  cp "$OUT_DIR/reward_net.pt" "$OUT_DIR/reward_net_final.pt" 2>/dev/null
  cp "$OUT_DIR/reward_net_best_eval.pt" "$OUT_DIR/reward_net.pt"
  echo ""
  echo "[prepare-airl] promoted reward_net_best_eval.pt -> reward_net.pt"
  echo "[prepare-airl] (the end-of-training version kept as reward_net_final.pt)"
fi
