#!/usr/bin/env bash
# Which part of the history does a TRAINED policy rely on?
#
#   docker exec -d sim /scripts/run_history_eval.sh
#
# The retrained ablation (run_history_ablation.sh) asks which history a policy
# can LEARN from, and its answer is blurred by training noise: three seeds with
# sd 4.2 left every unpaired interval containing zero. This asks the narrower
# question that needs no retraining -- which history a policy already trained
# on the full stack RELIES on -- by corrupting part of the stack at evaluation
# time, exactly as the blank-LiDAR test corrupts the scan.
#
# Every comparison therefore uses the SAME four checkpoints as the unmodified
# runs, so training noise cancels instead of being averaged over, and the
# episodes are the same pinned goal sequence. Pairing is by episode within seed.
#
#   self   past scans overwritten with the current scan; self-motion intact
#   scan   past self-motion overwritten with the current value; scans intact
#   none   both destroyed -- the total the policy gets from any history
#
# Reference, the same checkpoints unmodified: 60.0, 56.3, 56.7, 55.3 (mean 57.1).
#
# The caveat is the blank-LiDAR caveat. A frozen stack is an input the policy
# never trained on, so a drop measures what it relies on, not what it learned,
# and the three cells are out of distribution to different degrees.
#
# FOUR SEEDS x THREE CELLS, FIXED BEFORE THE RUNS.
set -uo pipefail
LOG=/logs/sweep/history_eval.log
STATUS=/logs/sweep/history_eval_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; EPISODES=300; GOAL_SEED=1000
OUT=/checkpoints/hist_eval

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

cfg_for () { case $1 in 0) echo /configs/bc_s5d_fs4.yaml;; *) echo "/configs/bc_s5d_fs4_seed$1.yaml";; esac; }
ckpt_for () { case $1 in 0) echo /checkpoints/s5d_fs4/bc_best.pt;; *) echo "/checkpoints/s5d_fs4_seed$1/bc_best.pt";; esac; }

for sd in 0 1 2 3; do
  [[ -f "$(ckpt_for $sd)" ]] || { warn "missing $(ckpt_for $sd)"; exit 1; }
  [[ -f "$(cfg_for $sd)"  ]] || { warn "missing $(cfg_for $sd)";  exit 1; }
done

say "checking the mask is a no-op for keep=all and alters the rest"
python3 /bc/verify_history_mask.py || { warn "mask verification"; exit 1; }

cool_gate
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet"; exit 1; }

say "evaluating 4 trained 4-frame policies under 3 corrupted histories"
for cell in self scan none; do for sd in 0 1 2 3; do
  out="$OUT/${cell}_seed${sd}.json"
  [[ -f "$out" ]] && { echo "  (exists) $(basename "$out")"; continue; }
  say "  keep=$cell seed=$sd"
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
      --config "$(cfg_for $sd)" --checkpoint "$(ckpt_for $sd)" \
      --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
      --history-at-eval "$cell" --out "$out" 2>&1 \
      | grep -E 'rewound|Result|MISMATCH|frozen|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $cell seed=$sd"
done; done

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "HISTORY EVAL COMPLETE"
python3 /scripts/history_eval_tests.py || true
