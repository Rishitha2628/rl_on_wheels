#!/usr/bin/env bash
# Re-run every BC evaluation with per-episode logging, so the pairing we set up
# can actually be used.
#
#   docker exec -d sim /scripts/run_paired.sh
#
# Every evaluation already rewinds the goal sequence, so episode i is the same
# goal in every run. Until now we recorded only the totals, which throws that
# away: comparing two totals is an unpaired test even when the episodes match.
# bc/eval.py now writes a per-episode success flag, which lets the analysis use
# McNemar's test on the episodes where two configurations disagree.
#
# The previous aggregate-only numbers are preserved in
# /checkpoints/aggregate_only/ for comparison; they should reproduce closely,
# since the only change is what gets written down.
set -uo pipefail
LOG=/logs/sweep/paired.log
STATUS=/logs/sweep/paired_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; MIN_FREE_GB=4; GOAL_SEED=1000

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }
room () { local f; f=$(df -BG --output=avail / | tail -1 | tr -dc '0-9'); (( f < MIN_FREE_GB )) && { warn "disk ${f}G"; return 1; }; return 0; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

has_per_episode () {   # skip only if the file ALREADY has per-episode data
  python3 -c "
import json,sys
try: sys.exit(0 if 'per_episode' in json.load(open(sys.argv[1])) else 1)
except Exception: sys.exit(1)" "$1" 2>/dev/null
}

ev () {   # config ckpt action_repeat overrides out episodes
  local cfg=$1 ck=$2 ar=$3 ovr=$4 out=$5 eps=${6:-300}
  has_per_episode "$out" && { echo "  (has per-episode) $(basename $out)"; return 0; }
  local flag="--no-overrides"; [[ "$ovr" == "1" ]] && flag=""
  rm -f "$out"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$eps" --action-repeat "$ar" $flag --quiet --out "$out" 2>&1 \
      | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $out"
}

say "PAIRED RE-RUN START"
while pgrep -f '[r]un_airl_s5.sh|[r]un_rate_matched.sh|[r]un_stage5_fixup.sh' > /dev/null; do sleep 120; done
room || exit 1

# ── stage 5 + DAgger: the factorial, the wrapper, and the floor ─────────────
say "STAGE 5 (+DAgger): factorial, wrapper, noise floor"
cool_gate
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet s5"; exit 1; }
for fs in 1 4; do for ar in 1 4; do
  say "  s5 fs=$fs ar=$ar"
  ev /configs/bc_s5d_fs${fs}.yaml /checkpoints/s5d_fs${fs}/bc_best.pt "$ar" 0 \
     /checkpoints/substrate_s5_dagger/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s5 wrapper"
ev /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt 1 0 /checkpoints/substrate_s5_dagger/bc_policy_alone.json
ev /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt 1 1 /checkpoints/substrate_s5_dagger/bc_with_overrides.json
say "  s5 noise floor"
for r in 1 2 3; do
  ev /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt 1 0 /checkpoints/floor_s5/r$r.json 100
done
say "  s5 rate-matched cells"
ev /configs/bc_s5r_fs1.yaml /checkpoints/s5r_fs1/bc_best.pt 4 0 /checkpoints/rate_matched/ratematched_fs1_2p5hz.json
ev /configs/bc_s5r_fs4.yaml /checkpoints/s5r_fs4/bc_best.pt 4 0 /checkpoints/rate_matched/ratematched_fs4_2p5hz.json
ev /configs/bc_s5v_fs4.yaml /checkpoints/s5v_fs4/bc_best.pt 1 0 /checkpoints/rate_matched/volctrl_fs4_10hz.json
stop_fleet

# ── stage 4 ─────────────────────────────────────────────────────────────────
say "STAGE 4: factorial, wrapper, noise floor"
cool_gate; room || exit 1
/scripts/launch_sim_fleet.sh -n 1 -s 4 --fast --goal-seed "$GOAL_SEED" || { warn "fleet s4"; exit 1; }
for fs in 1 4; do for ar in 1 4; do
  say "  s4 fs=$fs ar=$ar"
  ev /configs/bc_fs${fs}.yaml /checkpoints/substrate/bc_fs${fs}.pt "$ar" 0 \
     /checkpoints/substrate/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s4 wrapper"
ev /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt 1 0 /checkpoints/substrate/bc_policy_alone.json
ev /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt 1 1 /checkpoints/substrate/bc_with_overrides.json
say "  s4 noise floor"
for r in 1 2 3; do
  ev /configs/bc_fs4.yaml /checkpoints/substrate/bc_fs4.pt 1 0 /checkpoints/floor/s4_$r.json 100
done
stop_fleet

# ── recover the AIRL seed lost to the hang ─────────────────────────────────
say "AIRL seed 2 re-evaluation (its 4 checkpoints survived the hang)"
cool_gate
/scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet airl"
timeout 7200 python3 /rl/eval_budget.py --run-dir /checkpoints/sweep_s5/airl_s5_seed2 \
    --config /configs/airl.yaml --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "airl seed2 eval"
stop_fleet

say "PAIRED RE-RUN COMPLETE"
