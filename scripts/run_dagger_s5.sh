#!/usr/bin/env bash
# DAgger rounds on STAGE 5: BC drives, Nav2 relabels every visited state.
#
#   docker exec -d sim /scripts/run_dagger_s5.sh [episodes]
#   SMOKE=1 docker exec sim /scripts/run_dagger_s5.sh 12     # pipeline check
#
# REAL-TIME PACING IS MANDATORY (fast:=false). dagger_collect.py's
# ExpertListener.fresh() measures the age of Nav2's last /cmd_vel_expert with
# time.time() -- wall clock -- against a 0.3 s window, while the environment
# advances in sim time. At RTF 1 that window is 3 env steps; at RTF 5.8 it is
# ~17, so stale expert actions would be accepted as fresh and paired with
# observations the robot had already left. No error, just corrupted labels.
set -uo pipefail

EPISODES="${1:-200}"
SMOKE="${SMOKE:-0}"
STAGE=5
CFG=/configs/bc_s5_fs4.yaml
CKPT=/checkpoints/s5_fs4/bc_best.pt
OUTDIR=/demos
LOG=/logs/sweep/dagger_s5.log

exec > >(stdbuf -oL tee -a "$LOG") 2>&1
say () { echo ""; echo "=== $(date '+%a %H:%M:%S')  $*"; }

set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; set -u
export PYTHONPATH=/:/rl:${PYTHONPATH:-}
export IGN_GAZEBO_RESOURCE_PATH=/ros2_ws/install/tb3_rl_bridge/share/tb3_rl_bridge/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}
export ROS_DOMAIN_ID=40 IGN_PARTITION=tb3env0
mkdir -p /logs/sweep

[[ -f "$CKPT" ]] || { echo "missing $CKPT"; exit 1; }

cleanup () {
  say "tearing down"
  for p in ${NAV:-} ${SIM:-}; do kill -TERM -"$p" 2>/dev/null; done
  sleep 4
  for p in ${NAV:-} ${SIM:-}; do kill -9 -"$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

say "launching stage-5 sim at REAL TIME (fast:=false)"
setsid ros2 launch tb3_rl_bridge bridge.launch.py stage:=$STAGE headless:=true \
    fast:=false > /logs/dagger_sim.log 2>&1 &
SIM=$!
for i in $(seq 1 60); do ros2 service list 2>/dev/null | grep -q '^/step$' && break; sleep 2; done
ros2 service list 2>/dev/null | grep -q '^/step$' || { echo "sim never advertised /step"; exit 1; }
say "sim ready"

say "launching Nav2 (dagger_mode: expert publishes to /cmd_vel_expert)"
setsid ros2 launch tb3_nav2 nav2_bringup.launch.py stage:=$STAGE use_sim_time:=true \
    dagger_mode:=true > /logs/dagger_nav2.log 2>&1 &
NAV=$!
for i in $(seq 1 60); do
  grep -q 'Managed nodes are active' /logs/dagger_nav2.log 2>/dev/null && break; sleep 2
done
grep -q 'Managed nodes are active' /logs/dagger_nav2.log || { echo "Nav2 never became active"; exit 1; }
say "Nav2 active"
sleep 8

# Is the expert actually publishing? Without this the collector silently
# records zero usable pairs and we would not find out for hours.
say "checking /cmd_vel_expert is live"
n=$(timeout 25 ros2 topic hz /cmd_vel_expert 2>&1 | grep -c 'average rate' || true)
if [[ "${n:-0}" -lt 1 ]]; then
  echo "WARNING: no /cmd_vel_expert traffic yet (Nav2 may need a goal first)"
fi

say "collecting ${EPISODES} episodes"
OUT="$OUTDIR/dagger_s5_iter2.npz"
[[ "$SMOKE" == "1" ]] && OUT="/logs/dagger_s5_smoke.npz"
timeout 43200 python3 /bc/dagger_collect.py --config "$CFG" --checkpoint "$CKPT" \
    --out "$OUT" --episodes "$EPISODES" 2>&1 | tail -20

if [[ -f "$OUT" ]]; then
  python3 - "$OUT" <<'PY'
import numpy as np, sys
d = np.load(sys.argv[1])
n = d["obs"].shape[0]
print(f"\n  collected {n:,} transitions, {len(np.unique(d['ep_id']))} episodes -> {sys.argv[1]}")
print(f"  action range lin [{d['action'][:,0].min():.3f},{d['action'][:,0].max():.3f}] "
      f"ang [{d['action'][:,1].min():.3f},{d['action'][:,1].max():.3f}]")
if n == 0: print("  *** ZERO PAIRS - expert signal never arrived fresh ***")
PY
else
  echo "*** no output written ***"
fi
say "DAgger collection done"
