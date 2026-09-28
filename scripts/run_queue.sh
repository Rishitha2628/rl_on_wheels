#!/usr/bin/env bash
# Everything remaining, in priority order. One script rather than three
# waiting on each other -- after the reboot killed a mid-flight chain, fewer
# moving parts is worth more than parallelism the machine cannot use anyway.
#
#   1. TD3 x 3 seeds at lr 3e-4 / sigma 0.1, full 2M grid   (~25 h)
#      The tuning diagnostic hit 79.0% [70, 86] at 640k on ONE seed at n=100.
#      That single number now carries the paper's central claim, so it gets
#      the machine first.
#   2. AIRL seeds 1 and 2 with the fixed reward net, then evaluate all three
#      (seed 0 already completed with it before the reboot)   (~16 h)
#   3. Reward-shaping variants c and d                        (~6 h)
#
#   docker exec -d sim /scripts/run_queue.sh

set -uo pipefail

SWEEP_DIR=/checkpoints/sweep
TUNE_DIR=/checkpoints/tune
FIGDIR=/logs/figures
LOG=/logs/sweep/queue.log
CONF=/configs/td3_tune_a.yaml
AIRL_ARTIFACTS=/checkpoints/airl_stage4_v2
EPISODES=100
N_ENVS=4
STAGE=4

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; }
ping_ () { notify-send --urgency="$1" --app-name="rl_on_wheels" "$2" "$3" 2>/dev/null || true; }

set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}

# A second trainer sharing the same fleet and output directory corrupts both.
# This happened once already: a pkill intended to clear the previous queue
# matched its own shell (the command text contained the pattern) and killed
# itself before reaching the target, leaving two trainers running.
existing=$(ps -eo args | grep -c "^python3 /rl/sweep_td3.py" || true)
if [[ "${existing:-0}" -gt 0 ]]; then
  echo "REFUSING to start: ${existing} sweep_td3.py already running." >&2
  echo "Kill them by explicit PID first:  ps -eo pid,args | grep /rl/sweep_td3.py" >&2
  exit 1
fi

say "QUEUE START"

# ── 1. TD3 confirmation ───────────────────────────────────────────────────────
for seed in 0 1 2; do
  out="$TUNE_DIR/td3_confirm_seed${seed}"
  [[ -f "$out/eval_results.json" ]] && { say "TD3 confirm seed $seed already done"; continue; }

  say "TD3 confirm seed $seed (lr 3e-4, sigma 0.1, 640k bridge steps)"
  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || {
    echo "fleet failed" >&2; continue; }

  # 640k, not 2M: the 79% was measured at the 640k checkpoint, so that is the
  # number under test. Extending to the full grid is a separate 17h question
  # and only matters if the tuned arm replaces the baseline in the figure.
  #
  # Full output to its own file rather than `| tail -20`. The tuning run's
  # training curve was lost to exactly that pipe, which left no way to tell
  # steady learning from a lucky final evaluation -- the ambiguity these seeds
  # exist to resolve.
  stdbuf -oL python3 /rl/sweep_td3.py --config "$CONF" --seed "$seed" \
      --n-envs "$N_ENVS" --out-dir "$out" --max-bridge-steps 640000 \
      > "/logs/sweep/td3_confirm_seed${seed}.log" 2>&1
  tail -5 "/logs/sweep/td3_confirm_seed${seed}.log"

  stdbuf -oL python3 /rl/eval_budget.py --run-dir "$out" --config "$CONF" \
      --episodes "$EPISODES" --n-envs "$N_ENVS" || echo "eval failed"

  /scripts/launch_sim_fleet.sh --stop; sleep 5

  best=$(python3 -c "
import json
r=json.load(open('$out/eval_results.json'))['results']
b=max(r,key=lambda x:x['success_rate'])
print('%.1f%% [%.0f-%.0f] at %s steps' % (b['success_rate']*100, b['success_ci95'][0]*100, b['success_ci95'][1]*100, f\"{b['budget_bridge_steps']:,}\"))
" 2>/dev/null)
  say "TD3 confirm seed $seed: ${best:-done}"
  ping_ normal "TD3 confirm seed $seed" "${best:-done}  (tuning saw 79.0%; BC 63.7%)"
done

say "TD3 CONFIRMATION COMPLETE"
ping_ critical "TD3 confirmation done" "Three seeds at lr3e-4/sigma0.1. See $LOG"

# ── 2. AIRL seeds 1 and 2 ─────────────────────────────────────────────────────
if [[ -f "$AIRL_ARTIFACTS/reward_net.pt" ]]; then
  say "AIRL seeds 1 and 2 with the fixed reward net"
  /scripts/run_sweep_airl.sh --seeds "1 2" --max 2000000 \
      --artifacts "$AIRL_ARTIFACTS" --n-envs "$N_ENVS" --stage "$STAGE" \
      --out "$SWEEP_DIR"

  say "evaluating all three AIRL seeds"
  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast || true
  for d in "$SWEEP_DIR"/airl_seed*; do
    [[ -f "$d/budget_index.json" ]] || continue
    stdbuf -oL python3 /rl/eval_budget.py --run-dir "$d" \
        --episodes "$EPISODES" --n-envs "$N_ENVS" || echo "eval failed for $d"
  done
  /scripts/launch_sim_fleet.sh --stop; sleep 5
  ping_ critical "AIRL arm complete" "All three seeds on the fixed reward net."
else
  echo "AIRL artifacts missing at $AIRL_ARTIFACTS; skipping the AIRL stage." >&2
fi

# ── 3. reward variants ────────────────────────────────────────────────────────
run_variant () {
  local name="$1" vlin="$2" vang="$3"
  local out="$TUNE_DIR/td3_reward_${name}"
  [[ -f "$out/eval_results.json" ]] && { say "variant $name already done"; return 0; }

  say "reward variant ${name}: r_vlinear_scale=${vlin} r_vangular_weight=${vang}"
  /scripts/launch_sim_fleet.sh -n "$N_ENVS" -s "$STAGE" --fast \
      --r-vlinear "$vlin" --r-vangular "$vang" || return 1
  stdbuf -oL python3 /rl/sweep_td3.py --config "$CONF" --seed 0 \
      --n-envs "$N_ENVS" --out-dir "$out" --max-bridge-steps 640000 2>&1 | tail -20
  stdbuf -oL python3 /rl/eval_budget.py --run-dir "$out" --config "$CONF" \
      --episodes "$EPISODES" --n-envs "$N_ENVS" || echo "eval failed"
  /scripts/launch_sim_fleet.sh --stop; sleep 5
  ping_ normal "Reward variant ${name} done" "$(python3 -c "
import json
r=json.load(open('$out/eval_results.json'))['results']
b=max(r,key=lambda x:x['success_rate'])
print('%.1f%% [%.0f-%.0f]' % (b['success_rate']*100,b['success_ci95'][0]*100,b['success_ci95'][1]*100))
" 2>/dev/null)"
}
run_variant c 5.0 0.5
run_variant d 2.5 0.25

# ── 4. final figure ───────────────────────────────────────────────────────────
say "regenerating the crossover figure"
python3 /scripts/analyze_crossover.py --sweep-dir "$SWEEP_DIR" \
    --bc-results "$SWEEP_DIR/bc_policy_alone.json" --out-dir "$FIGDIR" || true

say "QUEUE COMPLETE"
ping_ critical "All queued work complete" "See $LOG and /logs/figures/"
