#!/usr/bin/env bash
# Three experiments isolating substrate effects, all on a fixed goal sequence.
#
#   1. Paired evaluation. Every method is scored on the same episodes
#      (goal_seed >= 0), which it previously was not -- reset_node seeded from
#      random_device, so part of any gap between two reported numbers was just
#      which goals each run happened to draw.
#   2. BC factorial over frame_stack {1,4} x action_repeat {1,4}. BC ran at
#      10 Hz on four stacked frames, TD3 at 2.5 Hz on one, so "BC vs TD3" mixed
#      the algorithm with the control rate and the observation history.
#   3. TD3 at action_repeat 1, three seeds, for the same reason from the other
#      side.
#
# Existing td3_tune_a checkpoints are re-evaluated on the fixed goals too, so
# the whole table is comparable.
#
#   docker exec -d sim /scripts/run_substrate.sh

set -uo pipefail

GOAL_SEED=1000            # every eval in this script uses seeds 1000..1003
EPISODES=300
# Number of parallel sims. Fewer means less heat and a longer run. It also
# changes the goal set: instance i uses seed GOAL_SEED+i, so results are only
# paired across runs that used the same N_ENVS.
N_ENVS=${N_ENVS:-4}
STAGE=4
BUDGET=640000
OUT=/checkpoints/substrate
LOG=/logs/sweep/substrate.log
MIN_FREE_GB=2             # abort rather than fill the disk

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; }

check_room () {
  local free
  free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  echo "  [disk] ${free}G free"
  if (( free < MIN_FREE_GB )); then
    echo "  [disk] below ${MIN_FREE_GB}G, stopping before anything is corrupted." >&2
    return 1
  fi
  return 0
}

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

if ps -eo args | grep -q '^python3 /rl/sweep_td3.py'; then
  echo "A trainer is already running; refusing to start a second." >&2
  exit 1
fi

say "SUBSTRATE RUNS START  (goal seeds ${GOAL_SEED}..$((GOAL_SEED + N_ENVS - 1)))"
check_room || exit 1

# ── 1+2. BC factorial on fixed goals ──────────────────────────────────────────
say "training BC at frame_stack 1 and 4"
for fs in 1 4; do
  ckpt="$OUT/bc_fs${fs}.pt"
  [[ -f "$ckpt" ]] && { echo "  fs=$fs already trained"; continue; }
  python3 -m bc.train --config "/configs/bc_fs${fs}.yaml" \
      --dataset /demos/stage4_dagger_agg2.npz 2>&1 | tail -3
  cp /checkpoints/bc_best.pt "$ckpt"
  echo "  saved $ckpt"
done

say "launching fleet on fixed goals"
/scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast --goal-seed "$GOAL_SEED" || exit 1

for fs in 1 4; do
  for ar in 1 4; do
    res="$OUT/bc_fs${fs}_ar${ar}.json"
    [[ -f "$res" ]] && { echo "  fs=$fs ar=$ar already evaluated"; continue; }
    say "BC frame_stack=${fs} action_repeat=${ar}"
    check_room || break 2
    # Domain ID, not the goal seed. ROS_DOMAIN_ID must be 0-232; passing the
    # seed (1000) here made every eval die instantly with "Calculated port
    # number is too high". The fleet puts instance 0 on DOMAIN_BASE.
    ROS_DOMAIN_ID=40 python3 /bc/eval.py \
        --config "/configs/bc_fs${fs}.yaml" \
        --checkpoint "$OUT/bc_fs${fs}.pt" \
        --episodes "$EPISODES" --action-repeat "$ar" \
        --no-overrides --quiet --out "$res" 2>&1 | tail -4
    if [[ ! -f "$res" ]]; then
      echo "  eval produced no output for fs=$fs ar=$ar; stopping rather than" >&2
      echo "  silently reporting a factorial with missing cells." >&2
      exit 1
    fi
  done
done

# ── re-evaluate the existing tuned TD3 on the same goals ──────────────────────
for s in 0 1 2; do
  d=/checkpoints/tune/td3_confirm_seed${s}
  [[ -d "$d" ]] || continue
  [[ -f "$d/eval_paired.json" ]] && continue
  say "re-evaluating td3_confirm_seed${s} on fixed goals"
  check_room || break
  python3 /rl/eval_budget.py --run-dir "$d" --config /configs/td3_tune_a.yaml \
      --episodes 100 --n-envs "$N_ENVS" --out "$d/eval_paired.json" 2>&1 | tail -3
done

/scripts/launch_sim_fleet.sh --stop; sleep 5

# ── 3. TD3 at action_repeat 1 ─────────────────────────────────────────────────
# Opt-in. This is the long part -- roughly 5-9 hours across three seeds, at
# sustained high CPU load -- whereas everything above it takes about an hour.
# The BC factorial already demonstrates the control-rate confound; this measures
# it from the RL side as well.
if [[ "${WITH_TD3:-0}" != "1" ]]; then
  say "skipping the TD3 action_repeat=1 runs (set WITH_TD3=1 to include them)"
  say "SUBSTRATE RUNS COMPLETE"
  python3 /scripts/summarize_substrate.py 2>/dev/null || true
  notify-send --urgency=critical --app-name="rl_on_wheels" \
    "BC factorial done" "See $LOG" 2>/dev/null || true
  exit 0
fi

for seed in 0 1 2; do
  d="$OUT/td3_ar1_seed${seed}"
  [[ -f "$d/eval_results.json" ]] && { echo "  ar1 seed $seed done"; continue; }
  say "TD3 action_repeat=1, seed ${seed}, ${BUDGET} bridge steps"
  check_room || break

  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast --goal-seed "$GOAL_SEED" || continue
  python3 /rl/sweep_td3.py --config /configs/td3_ar1.yaml --seed "$seed" \
      --n-envs "$N_ENVS" --out-dir "$d" --max-bridge-steps "$BUDGET" \
      > "/logs/sweep/td3_ar1_seed${seed}.log" 2>&1
  tail -3 "/logs/sweep/td3_ar1_seed${seed}.log"

  python3 /rl/eval_budget.py --run-dir "$d" --config /configs/td3_ar1.yaml \
      --episodes 100 --n-envs "$N_ENVS" 2>&1 | tail -3
  /scripts/launch_sim_fleet.sh --stop; sleep 5
  check_room || break
done

say "SUBSTRATE RUNS COMPLETE"
python3 /scripts/summarize_substrate.py 2>/dev/null || true
notify-send --urgency=critical --app-name="rl_on_wheels" \
  "Substrate runs done" "See $LOG" 2>/dev/null || true
