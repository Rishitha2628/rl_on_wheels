#!/usr/bin/env bash
# Run the AIRL phase-2 arm of the interaction-budget sweep.
#
# Deliberately a separate file from run_sweep.sh rather than a flag on it.
# run_sweep.sh is typically mid-flight when this gets written, and bash reads
# a script lazily by byte offset -- editing a running script makes it resume
# at the wrong place and execute garbage. The arms also take different
# arguments (this one needs a reward net and a warm-start policy), so they do
# not share much beyond the seed loop.
#
# Prerequisites, produced by scripts/prepare_airl.sh:
#   <artifact-dir>/reward_net.pt
#   <artifact-dir>/policy_after_bc.zip
#
# Usage:
#   scripts/run_sweep_airl.sh                  # seeds 0 1 2, full 4M budget
#   scripts/run_sweep_airl.sh --smoke          # 60k budget, seed 0
#   scripts/run_sweep_airl.sh --seeds "0 1"

set -uo pipefail

SEEDS="0 1 2"
N_ENVS=4
STAGE=4
MAX_BRIDGE=4000000
ARTIFACT_DIR=/checkpoints/airl_stage4
OUT_ROOT=/checkpoints/sweep
FRAME_STACK=1
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)        SEEDS="$2"; shift 2 ;;
    --n-envs)       N_ENVS="$2"; shift 2 ;;
    --stage)        STAGE="$2"; shift 2 ;;
    --max)          MAX_BRIDGE="$2"; shift 2 ;;
    --artifacts)    ARTIFACT_DIR="$2"; shift 2 ;;
    --out)          OUT_ROOT="$2"; shift 2 ;;
    --frame-stack)  FRAME_STACK="$2"; shift 2 ;;
    --smoke)        SMOKE=1; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then
  SEEDS="0"
  MAX_BRIDGE=60000
  OUT_ROOT=/checkpoints/sweep_smoke
  echo "[airl-sweep] SMOKE MODE -- ${MAX_BRIDGE} bridge steps, seed 0 only"
fi

REWARD_NET="$ARTIFACT_DIR/reward_net.pt"
INIT_POLICY="$ARTIFACT_DIR/policy_after_bc.zip"

# ── refuse to trample the TD3 arm ─────────────────────────────────────────────
if pgrep -f sweep_td3.py >/dev/null 2>&1; then
  echo "[airl-sweep] REFUSING to start: the TD3 sweep is still running." >&2
  echo "[airl-sweep] Both saturate the same 4-instance fleet; running them" >&2
  echo "[airl-sweep] together halves each one's throughput and invalidates" >&2
  echo "[airl-sweep] the wall-clock timings the paper reports." >&2
  exit 1
fi

# ── prerequisites ─────────────────────────────────────────────────────────────
missing=0
for f in "$REWARD_NET" "$INIT_POLICY"; do
  if [[ ! -f "$f" ]]; then echo "[airl-sweep] MISSING $f" >&2; missing=1; fi
done
if [[ $missing -eq 1 ]]; then
  echo "" >&2
  echo "[airl-sweep] These are not in git -- .gitignore excludes *.zip, so the" >&2
  echo "[airl-sweep] originals were never committed. Regenerate them from the" >&2
  echo "[airl-sweep] surviving Nav2 demos:" >&2
  echo "" >&2
  echo "    scripts/prepare_airl.sh --out-dir $ARTIFACT_DIR" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

mkdir -p "$OUT_ROOT" /logs/sweep

echo "[airl-sweep] reward net:   $REWARD_NET"
echo "[airl-sweep] warm start:   $INIT_POLICY"
echo "[airl-sweep] frame stack:  $FRAME_STACK"
echo "[airl-sweep] seeds:        $SEEDS"

for seed in $SEEDS; do
  echo ""
  echo "=================================================================="
  echo "[airl-sweep] seed $seed  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=================================================================="

  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
    echo "[airl-sweep] fleet failed to start for seed $seed; skipping" >&2
    continue
  }

  out_dir="$OUT_ROOT/airl_seed${seed}"
  log="/logs/sweep/airl_seed${seed}.log"

  stdbuf -oL -eL python3 /airl/sweep_airl.py \
      --env-config /configs/td3.yaml \
      --config /configs/airl.yaml \
      --reward-net "$REWARD_NET" \
      --init-policy "$INIT_POLICY" \
      --frame-stack "$FRAME_STACK" \
      --seed "$seed" \
      --n-envs "$N_ENVS" \
      --out-dir "$out_dir" \
      --max-bridge-steps "$MAX_BRIDGE" \
    2>&1 | tee "$log"

  status=${PIPESTATUS[0]}
  echo "[airl-sweep] seed $seed exited with status $status"

  /scripts/launch_sim_fleet.sh --stop
  sleep 5
done

echo ""
echo "[airl-sweep] all seeds done. Checkpoints under $OUT_ROOT"
for seed in $SEEDS; do
  s="$OUT_ROOT/airl_seed${seed}/run_summary.json"
  [[ -f "$s" ]] && echo "  seed $seed: $(python3 -c "
import json
d=json.load(open('$s'))
print(f\"{d['wall_total_h']} h, budgets {d['budgets_reached']}\")
" 2>/dev/null)"
done
