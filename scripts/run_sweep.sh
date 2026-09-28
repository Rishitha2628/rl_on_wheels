#!/usr/bin/env bash
# Run the TD3 arm of the interaction-budget sweep: N seeds, sequentially,
# each on a freshly launched 4-instance fleet.
#
# Sequential rather than parallel on purpose -- the machine already saturates
# at four sim instances, so running two seeds at once would halve each one's
# throughput and change the update-to-data timing rather than finish sooner.
#
# The fleet is restarted between seeds. Ignition leaks memory slowly over
# millions of steps, and a fresh fleet per seed keeps runs independent.
#
# Usage:
#   scripts/run_sweep.sh                    # seeds 0 1 2, full 4M budget
#   scripts/run_sweep.sh --smoke            # 60k budget, one seed -- verify wiring
#   scripts/run_sweep.sh --seeds "0 1"      # subset

set -uo pipefail

SEEDS="0 1 2"
N_ENVS=4
STAGE=4
MAX_BRIDGE=4000000
OUT_ROOT=/checkpoints/sweep
CONFIG=/configs/td3.yaml
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)   SEEDS="$2"; shift 2 ;;
    --n-envs)  N_ENVS="$2"; shift 2 ;;
    --stage)   STAGE="$2"; shift 2 ;;
    --max)     MAX_BRIDGE="$2"; shift 2 ;;
    --out)     OUT_ROOT="$2"; shift 2 ;;
    --smoke)   SMOKE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then
  SEEDS="0"
  MAX_BRIDGE=60000
  OUT_ROOT=/checkpoints/sweep_smoke
  echo "[sweep] SMOKE MODE -- ${MAX_BRIDGE} bridge steps, seed 0 only"
fi

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/rl:/:${PYTHONPATH:-}

mkdir -p "$OUT_ROOT" /logs/sweep

for seed in $SEEDS; do
  echo ""
  echo "=================================================================="
  echo "[sweep] seed $seed  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=================================================================="

  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
    echo "[sweep] fleet failed to start for seed $seed; skipping" >&2
    continue
  }

  out_dir="$OUT_ROOT/td3_seed${seed}"
  log="/logs/sweep/td3_seed${seed}.log"

  # stdbuf so the log is followable live rather than appearing in 4 KB bursts.
  stdbuf -oL -eL python3 /rl/sweep_td3.py \
      --config "$CONFIG" \
      --seed "$seed" \
      --n-envs "$N_ENVS" \
      --out-dir "$out_dir" \
      --max-bridge-steps "$MAX_BRIDGE" \
    2>&1 | tee "$log"

  status=${PIPESTATUS[0]}
  echo "[sweep] seed $seed exited with status $status"

  /scripts/launch_sim_fleet.sh --stop
  sleep 5
done

echo ""
echo "[sweep] all seeds done. Checkpoints under $OUT_ROOT"
for seed in $SEEDS; do
  s="$OUT_ROOT/td3_seed${seed}/run_summary.json"
  [[ -f "$s" ]] && echo "  seed $seed: $(python3 -c "
import json,sys
d=json.load(open('$s'))
print(f\"{d['wall_total_h']} h, budgets {d['budgets_reached']}, {d['mean_bridge_steps_per_sec']} steps/s\")
" 2>/dev/null)"
done
