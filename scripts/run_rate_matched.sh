#!/usr/bin/env bash
# Does the rate x frame-stack interaction survive when each cell is TRAINED at
# its own control rate?
#
# The factorial so far trains every policy on 10 Hz demonstrations and only
# evaluates at 2.5 Hz. With four stacked frames that means 0.1 s spacing in
# training and 0.4 s at evaluation -- a distribution the network never saw. So
# "4 frames + slow control is terrible" may just be distance from the training
# conditions. Here each policy is trained on demonstrations resampled to its own
# rate, so the spacing matches in both.
#
# Three new cells:
#   s5r_fs1 @ ar=4   1 frame,  trained and evaluated at 2.5 Hz
#   s5r_fs4 @ ar=4   4 frames, trained and evaluated at 2.5 Hz   <- the decider
#   s5v_fs4 @ ar=1   4 frames, 10 Hz, volume-matched to the 2.5 Hz set
#
# The volume control exists because resampling to 2.5 Hz leaves 4x fewer
# transitions; without it, a drop could be blamed on data rather than rate.
set -uo pipefail
LOG=/logs/sweep/rate_matched.log
STATUS=/logs/sweep/rate_matched_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; EPISODES=300; GOAL_SEED=1000
OUT=/checkpoints/rate_matched

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

say "waiting for the fixup and AIRL queues"
while pgrep -f '[r]un_stage5_fixup.sh|[r]un_airl_s5.sh' > /dev/null; do sleep 180; done
say "clear - starting rate-matched cells"

for c in s5r_fs1 s5r_fs4 s5v_fs4; do
  [[ -f /checkpoints/$c/bc_best.pt ]] || { warn "missing /checkpoints/$c/bc_best.pt"; exit 1; }
done

cool_gate
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet"; exit 1; }

run_cell () {   # ckpt_dir config action_repeat out_name
  local ck=$1 cfg=$2 ar=$3 out="$OUT/$4.json"
  [[ -f "$out" ]] && { echo "  (exists) $out"; return 0; }
  say "  $4  (ckpt $ck, ar=$ar)"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config "/configs/${cfg}.yaml" --checkpoint "/checkpoints/${ck}/bc_best.pt" \
      --episodes "$EPISODES" --action-repeat "$ar" --no-overrides --quiet \
      --out "$out" 2>&1 | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $4"
}

run_cell s5r_fs1 bc_s5r_fs1 4 ratematched_fs1_2p5hz
run_cell s5r_fs4 bc_s5r_fs4 4 ratematched_fs4_2p5hz
run_cell s5v_fs4 bc_s5v_fs4 1 volctrl_fs4_10hz

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "RATE-MATCHED CELLS COMPLETE"
python3 - <<'PY' || true
import json, glob, math, os
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*max(0,c-h),100*min(1,c+h)
for f in sorted(glob.glob("/checkpoints/rate_matched/*.json")):
    d=json.load(open(f)); k,n=d["successes"],d["episodes"]; lo,hi=wilson(k,n)
    print(f"  {os.path.basename(f)[:-5]:26s} {k:3d}/{n} = {100*k/n:5.1f}%  [{lo:4.1f},{hi:4.1f}]")
PY
