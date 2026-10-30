#!/usr/bin/env bash
# Re-runs the three steps that failed because /configs is mounted read-only in
# the container: the BC retrain on DAgger data (done separately on the host side),
# the stage-5 noise floor and factorial on the DAgger policy, and the TD3
# shipped-config arm. The configs now exist, written from the host.
#
#   docker exec -d sim /scripts/run_stage5_fixup.sh
#
# Waits for any other queue to finish first so the two never share a fleet.
set -uo pipefail
LOG=/logs/sweep/stage5_fixup.log
STATUS=/logs/sweep/stage5_fixup_status.txt
GO_TEMP=${GO_TEMP:-93}; MAX_WAIT=1800; MIN_FREE_GB=4; EPISODES=300; GOAL_SEED=1000

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say  () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
warn () { echo "!!! $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  FAILED: $*" >> "$STATUS"; }
cpu_c () { local t; t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1); echo $(( ${t:-0} / 1000 )); }
cool_gate () { local w=0 t; t=$(cpu_c); while (( t > GO_TEMP && w < MAX_WAIT )); do echo "  [thermal] ${t}C, waiting"; sleep 60; w=$((w+60)); t=$(cpu_c); done; echo "  [thermal] start ${t}C"; }
room () { local f; f=$(df -BG --output=avail / | tail -1 | tr -dc '0-9'); (( f < MIN_FREE_GB )) && { warn "disk ${f}G"; return 1; }; return 0; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
mkdir -p /logs/sweep /checkpoints/floor_s5 /checkpoints/substrate_s5_dagger
stop_fleet () { /scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4; }

say "waiting for the main stage-5 queue to finish"
while pgrep -f '[r]un_stage5_all.sh' > /dev/null; do sleep 120; done
say "main queue done - starting fixup"

for fs in 1 4; do
  [[ -f /checkpoints/s5d_fs${fs}/bc_best.pt ]] || { warn "missing s5d_fs${fs} policy - run bc.train first"; exit 1; }
done

bc_eval () {
  local cfg=$1 ck=$2 ar=$3 ovr=$4 out=$5
  [[ -f "$out" ]] && { echo "  (exists) $out"; return 0; }
  local flag="--no-overrides"; [[ "$ovr" == "1" ]] && flag=""
  ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py --config "$cfg" --checkpoint "$ck" \
      --episodes "$EPISODES" --action-repeat "$ar" $flag --quiet --out "$out" 2>&1 \
      | grep -E 'rewound|Result|WARNING' || true
  [[ -f "$out" ]] || warn "no output: $out"
}

say "STEP A  stage-5 noise floor on the DAgger policy (3 x 100)"
cool_gate; room || exit 1
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet"
for r in 1 2 3; do
  [[ -f /checkpoints/floor_s5/r$r.json ]] && continue
  ROS_DOMAIN_ID=40 timeout 3600 python3 /bc/eval.py --config /configs/bc_s5d_fs4.yaml \
      --checkpoint /checkpoints/s5d_fs4/bc_best.pt --episodes 100 --action-repeat 1 \
      --no-overrides --quiet --out /checkpoints/floor_s5/r$r.json 2>&1 | grep -E 'rewound|Result' || true
done
stop_fleet

say "STEP B  stage-5 BC factorial + wrapper on the DAgger policy (6 x ${EPISODES})"
cool_gate; room || exit 1
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || warn "fleet"
for fs in 1 4; do for ar in 1 4; do
  say "  s5d fs=$fs ar=$ar"
  bc_eval /configs/bc_s5d_fs${fs}.yaml /checkpoints/s5d_fs${fs}/bc_best.pt "$ar" 0 \
          /checkpoints/substrate_s5_dagger/bc_fs${fs}_ar${ar}.json
  room || break 2
done; done
say "  s5d wrapper on/off"
bc_eval /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt 1 0 /checkpoints/substrate_s5_dagger/bc_policy_alone.json
bc_eval /configs/bc_s5d_fs4.yaml /checkpoints/s5d_fs4/bc_best.pt 1 1 /checkpoints/substrate_s5_dagger/bc_with_overrides.json
stop_fleet

say "STEP C  stage-5 TD3 shipped config, 3 seeds (~8 h)"
for seed in 0 1 2; do
  d=/checkpoints/tune_s5/td3_s5_shipped_seed${seed}
  [[ -f "$d/eval_results.json" ]] && { echo "  seed $seed done"; continue; }
  cool_gate; room || break
  say "  TD3 shipped seed $seed"
  /scripts/launch_sim_fleet.sh -n 4 -s 5 --fast --goal-seed "$GOAL_SEED" || { warn "fleet seed $seed"; continue; }
  timeout 54000 python3 /rl/sweep_td3.py --config /configs/td3_s5_shipped.yaml --seed "$seed" \
      --n-envs 4 --out-dir "$d" --max-bridge-steps 640000 \
      > "/logs/sweep/td3_s5_shipped_seed${seed}.log" 2>&1 || warn "train seed $seed"
  timeout 10800 python3 /rl/eval_budget.py --run-dir "$d" --config /configs/td3_s5_shipped.yaml \
      --episodes 100 --n-envs 4 2>&1 | tail -3 || warn "eval seed $seed"
  stop_fleet
done
say "FIXUP COMPLETE"
