#!/usr/bin/env bash
# Unattended overnight queue, ordered highest-value / lowest-risk first.
#
#   docker exec -d sim /scripts/run_overnight.sh
#
# Every step is independent: a failure is logged and the queue moves on, so one
# bad step cannot cost the whole night. Steps skip themselves if their output
# already exists, so the script is safe to re-run.
#
# All of this runs AFTER the /reset_goal_sequence fix, so every evaluation
# starts at goal 1 of its seeded sequence. Pre-fix results are in
# /checkpoints/prefix_* and are NOT comparable with these.

set -uo pipefail

LOG=/logs/sweep/overnight.log
STATUS=/logs/sweep/overnight_status.txt
MIN_FREE_GB=4
EPISODES=300
GOAL_SEED=1000

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*" ; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }

room () {
  local free; free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  (( free < MIN_FREE_GB )) && { warn "disk below ${MIN_FREE_GB}G (${free}G) - stopping queue"; return 1; }
  return 0
}

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p /logs/sweep /checkpoints/floor

stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

# bc_eval <config> <ckpt> <action_repeat> <overrides:0|1> <out>
bc_eval () {
  local cfg=$1 ck=$2 ar=$3 ovr=$4 out=$5
  [[ -f "$out" ]] && { echo "  (exists, skipping) $out"; return 0; }
  local flag="--no-overrides"; [[ "$ovr" == "1" ]] && flag=""
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat "$ar" $flag --quiet --out "$out" 2>&1 \
      | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $out"
}

say "OVERNIGHT QUEUE START  (goal fix active: /reset_goal_sequence)"
room || exit 1

# ── 1. stage-4 noise floor: does the 8-point stage-4 span clear it? ───────────
say "STEP 1/5  stage-4 noise floor (3 x 100 episodes, identical config)"
if [[ ! -f /checkpoints/floor/s4_3.json ]]; then
  /scripts/launch_sim_fleet.sh -n 1 -s 4 --fast --goal-seed "$GOAL_SEED" || warn "fleet (s4 floor)"
  for r in 1 2 3; do
    [[ -f /checkpoints/floor/s4_$r.json ]] && continue
    ROS_DOMAIN_ID=40 timeout 3600 python3 /bc/eval.py --config /configs/bc_fs4.yaml \
        --checkpoint /checkpoints/substrate/bc_fs4.pt --episodes 100 --action-repeat 1 \
        --no-overrides --quiet --out /checkpoints/floor/s4_$r.json 2>&1 \
        | grep -E 'rewound|Result' || true
  done
  stop_fleet
fi
room || exit 1

# ── 2. stage-4 factorial + wrapper, goal-paired ──────────────────────────────
say "STEP 2/5  stage-4 factorial + wrapper, 6 x ${EPISODES} episodes"
/scripts/launch_sim_fleet.sh -n 1 -s 4 --fast --goal-seed "$GOAL_SEED" || warn "fleet (s4)"
for fs in 1 4; do for ar in 1 4; do
  say "  s4 fs=$fs ar=$ar"
  bc_eval /configs/bc_fs${fs}.yaml /checkpoints/substrate/bc_fs${fs}.pt "$ar" 0 \
          /checkpoints/substrate/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s4 wrapper on/off"
bc_eval /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt 1 0 /checkpoints/substrate/bc_policy_alone.json
bc_eval /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt 1 1 /checkpoints/substrate/bc_with_overrides.json
stop_fleet
room || exit 1

# ── 3. stage-5 factorial + wrapper, goal-paired ──────────────────────────────
say "STEP 3/5  stage-5 factorial + wrapper, 6 x ${EPISODES} episodes"
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet (s5)"
for fs in 1 4; do for ar in 1 4; do
  say "  s5 fs=$fs ar=$ar"
  bc_eval /configs/bc_s5_fs${fs}.yaml /checkpoints/s5_fs${fs}/bc_best.pt "$ar" 0 \
          /checkpoints/substrate_s5/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s5 wrapper on/off"
bc_eval /configs/bc_s5_fs4.yaml /checkpoints/s5_fs4/bc_best.pt 1 0 /checkpoints/substrate_s5/bc_policy_alone.json
bc_eval /configs/bc_s5_fs4.yaml /checkpoints/s5_fs4/bc_best.pt 1 1 /checkpoints/substrate_s5/bc_with_overrides.json
stop_fleet
room || exit 1

# ── 4. stage-5 TD3, 3 seeds to 640k ──────────────────────────────────────────
# No Nav2 needed, so it is safe to run unattended -- unlike DAgger, which needs
# the expert stack live and has never been run end-to-end in this session.
say "STEP 4/5  stage-5 TD3, 3 seeds to 640k bridge steps (~8 h)"
for seed in 0 1 2; do
  d=/checkpoints/tune_s5/td3_s5_seed${seed}
  [[ -f "$d/eval_results.json" ]] && { echo "  seed $seed done"; continue; }
  room || break
  say "  TD3 stage-5 seed $seed"
  /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet (td3 s5 seed $seed)"; continue; }
  timeout 54000 python3 /rl/sweep_td3.py --config /configs/td3_s5.yaml --seed "$seed" \
      --n-envs 4 --out-dir "$d" --max-bridge-steps 640000 \
      > "/logs/sweep/td3_s5_seed${seed}.log" 2>&1 || warn "training seed $seed"
  tail -3 "/logs/sweep/td3_s5_seed${seed}.log"
  timeout 10800 python3 /rl/eval_budget.py --run-dir "$d" --config /configs/td3_s5.yaml \
      --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "eval seed $seed"
  stop_fleet
done

# ── 5. summary ───────────────────────────────────────────────────────────────
say "STEP 5/5  summary"
python3 - <<'PY' || true
import json, glob, math, os
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)
for label, pat in (("stage-4 floor", "/checkpoints/floor/s4_*.json"),
                   ("stage 4", "/checkpoints/substrate/*.json"),
                   ("stage 5", "/checkpoints/substrate_s5/*.json")):
    fs = sorted(glob.glob(pat))
    if not fs: continue
    print(f"\n--- {label} ---")
    for f in fs:
        d = json.load(open(f)); k, n = d["successes"], d["episodes"]
        lo, hi = wilson(k, n)
        print(f"  {os.path.basename(f)[:-5]:22s} {k:3d}/{n} = {100*k/n:5.1f}%  [{100*lo:4.1f},{100*hi:4.1f}]")
PY
say "OVERNIGHT QUEUE COMPLETE"
