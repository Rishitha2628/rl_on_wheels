#!/usr/bin/env bash
# Sensor and actuator interventions, applied at EVALUATION time to already
# trained stage-5 DAgger policies.
#
#   docker exec -d sim /scripts/run_substrate_sensor.sh
#
# Waits for run_static_obs.sh to finish first -- one sim instance, and two
# concurrent evals would contend for CPU and corrupt the wall-clock pacing.
#
# What this measures: the robustness of a FIXED policy to a degraded
# substrate. It does not measure whether training under these conditions
# helps. Both factors are run for 1-frame and 4-frame policies, because a
# frame history is a plausible noise filter -- if stacking's benefit grows
# under sensor noise, that is evidence for what the stack is actually doing.
#
# Reference points (same protocol, undegraded): fs1 50.7%, fs4 60.0%
# from checkpoints/substrate_s5_dagger/bc_fs{1,4}_ar1.json.
#
# Sim lidar is 6.0 m with 1 cm Gaussian noise already; the real LDS-01 is
# 3.5 m with several cm of noise and dropouts, so these move toward reality.
set -uo pipefail
LOG=/logs/sweep/substrate_sensor.log
STATUS=/logs/sweep/substrate_sensor_status.txt
GOAL_SEED=1000; EPISODES=300; OUT=/checkpoints/substrate_sensor
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; echo "$(date '+%H:%M')  $*" >> "$STATUS"; }
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/bc:${PYTHONPATH:-}
mkdir -p "$OUT" /logs/sweep

# ── wait for the static-obstacle run to clear ───────────────────────────────
for _ in $(seq 1 480); do
  pgrep -f 'run_static_obs' >/dev/null 2>&1 || break
  sleep 30
done
if pgrep -f 'run_static_obs' >/dev/null 2>&1; then
  echo "ABORT: run_static_obs still running after 4h" >&2; exit 1
fi
say "static run clear; starting sensor/actuator sweep"

/scripts/launch_sim_fleet.sh --stop >/dev/null 2>&1; sleep 4
/scripts/launch_sim_fleet.sh -n 1 -s 5 --fast --goal-seed "$GOAL_SEED" || exit 1

# name:flags  -- one condition per line
CONDS=(
  "noise003:--lidar-noise 0.03"
  "noise010:--lidar-noise 0.10"
  "drop010:--lidar-dropout 0.10"
  "range35:--lidar-max-range 3.5"
  "tau030:--motor-tau 0.3"
)

for fs in 1 4; do
  cfg=/configs/bc_s5d_fs${fs}.yaml
  ck=/checkpoints/s5d_fs${fs}/bc_best.pt
  for spec in "${CONDS[@]}"; do
    name=${spec%%:*}; flags=${spec#*:}
    out="$OUT/fs${fs}_${name}.json"
    [[ -f "$out" ]] && { echo "  (exists) fs=$fs $name"; continue; }
    say "  fs=$fs  $name  [$flags]"
    # shellcheck disable=SC2086
    ROS_DOMAIN_ID=40 timeout 7200 python3 /bc/eval.py \
        --config "$cfg" --checkpoint "$ck" \
        --episodes "$EPISODES" --action-repeat 1 --no-overrides --quiet \
        $flags --out "$out" 2>&1 | grep -E 'substrate interventions|rewound|Result' || true
  done
done

/scripts/launch_sim_fleet.sh --stop; sleep 4
say "SENSOR/ACTUATOR SWEEP COMPLETE"
