#!/usr/bin/env bash
# Run N DAgger iterations against an already-running sim + shadow Nav2.
#
# Each iteration:
#   1. collect (BC drives, Nav2 relabels) → dagger_iterN.npz
#   2. merge   (base_demos + all dagger iters so far) → aggN.npz
#   3. train   (full aggregated dataset)              → bc_best.pt
#
# Pre-requirements (start in separate terminals BEFORE running this):
#   - ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 dynamic_obstacles:=true
#   - ros2 launch tb3_nav2 nav2_bringup.launch.py stage:=4 dagger_mode:=true
#
# Usage (inside container):
#   bash /bc/dagger_iterate.sh 3 200   # 3 iterations × 200 episodes each
#
set -euo pipefail

N_ITERS="${1:-3}"
EPISODES_PER_ITER="${2:-200}"
BASE_DEMOS="${BASE_DEMOS:-/demos/stage4_bc.npz}"
DEMOS_DIR="${DEMOS_DIR:-/demos}"
CKPT="${CKPT:-/checkpoints/bc_best.pt}"
CONFIG="${CONFIG:-/configs/bc.yaml}"

if [[ ! -f "${BASE_DEMOS}" ]]; then
  echo "ERROR: base demos NPZ not found at ${BASE_DEMOS}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: starting checkpoint not found at ${CKPT}" >&2
  exit 1
fi

INPUTS=("${BASE_DEMOS}")

for ((i = 1; i <= N_ITERS; i++)); do
  echo "════════════════════════════════════════════════════════════════"
  echo "  DAgger iteration ${i}/${N_ITERS}"
  echo "════════════════════════════════════════════════════════════════"

  ITER_NPZ="${DEMOS_DIR}/dagger_iter${i}.npz"
  AGG_NPZ="${DEMOS_DIR}/stage4_dagger_agg${i}.npz"

  echo "[${i}/${N_ITERS}] collecting (BC drives, Nav2 relabels) → ${ITER_NPZ}"
  PYTHONPATH=/:${PYTHONPATH:-} python3 /bc/dagger_collect.py \
    --config     "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --out        "${ITER_NPZ}" \
    --episodes   "${EPISODES_PER_ITER}"

  INPUTS+=("${ITER_NPZ}")

  echo "[${i}/${N_ITERS}] merging ${#INPUTS[@]} NPZs → ${AGG_NPZ}"
  PYTHONPATH=/:${PYTHONPATH:-} python3 /bc/dataset_merge.py \
    --inputs "${INPUTS[@]}" \
    --out    "${AGG_NPZ}"

  echo "[${i}/${N_ITERS}] training on ${AGG_NPZ}"
  PYTHONPATH=/:${PYTHONPATH:-} python3 -m bc.train \
    --config  "${CONFIG}" \
    --dataset "${AGG_NPZ}"
  # train.py overwrites /checkpoints/bc_best.pt automatically when val improves.
done

echo ""
echo "Done. Final checkpoint: ${CKPT}"
echo "Evaluate with:"
echo "  PYTHONPATH=/:${PYTHONPATH:-} python3 /bc/eval.py --checkpoint ${CKPT} --episodes 20"
