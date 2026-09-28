#!/usr/bin/env bash
# Launch N isolated sim instances for vectorised training.
#
# Each instance needs isolation on TWO transports, not one:
#
#   ROS_DOMAIN_ID  separates the DDS graph, so /step, /scan, /cmd_vel and the
#                  bridge services of instance i are invisible to instance j.
#
#   IGN_PARTITION  separates Ignition's own transport. This is NOT redundant:
#                  env_bridge_node subscribes to /world/empty/dynamic_pose/info
#                  over gz-transport directly, and reset_node calls the
#                  Ignition set_pose/create services. Without a partition every
#                  instance would teleport every other instance's robot.
#
# Usage:
#   scripts/launch_sim_fleet.sh -n 4 -s 4 --fast      # 4 accelerated sims
#   scripts/launch_sim_fleet.sh -n 1 -s 4 --rt        # 1 real-time sim
#   scripts/launch_sim_fleet.sh --stop                # tear the fleet down
#
# Instance i is reachable at ROS_DOMAIN_ID = DOMAIN_BASE + i.

set -uo pipefail

N=4
STAGE=4
FAST=true
DOMAIN_BASE=40
# Reward shaping weights, passed through to every instance. Defaults match the
# original reward exactly.
R_VLINEAR_SCALE=10.0
R_VANGULAR_WEIGHT=1.0
FLEET_DIR=/logs/fleet
READY_TIMEOUT=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num)        N="$2"; shift 2 ;;
    -s|--stage)      STAGE="$2"; shift 2 ;;
    --fast)          FAST=true; shift ;;
    --rt)            FAST=false; shift ;;
    --domain-base)   DOMAIN_BASE="$2"; shift 2 ;;
    --r-vlinear)     R_VLINEAR_SCALE="$2"; shift 2 ;;
    --r-vangular)    R_VANGULAR_WEIGHT="$2"; shift 2 ;;
    --stop)          STOP=1; shift ;;
    -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ── teardown ──────────────────────────────────────────────────────────────────
stop_fleet() {
  echo "[fleet] stopping..."
  if [[ -f "$FLEET_DIR/pids" ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && kill -TERM "-$pid" 2>/dev/null
    done < "$FLEET_DIR/pids"
    sleep 3
    while read -r pid; do
      [[ -n "$pid" ]] && kill -9 "-$pid" 2>/dev/null
    done < "$FLEET_DIR/pids"
    rm -f "$FLEET_DIR/pids"
  fi
  # Belt and braces -- ros2 launch does not always reap its children.
  pkill -9 -f "ign gazebo"      2>/dev/null
  pkill -9 -f env_bridge_node   2>/dev/null
  pkill -9 -f reset_node        2>/dev/null
  pkill -9 -f dynamic_obstacle  2>/dev/null
  pkill -9 -f parameter_bridge  2>/dev/null
  sleep 2
  echo "[fleet] stopped."
}

if [[ -n "${STOP:-}" ]]; then
  stop_fleet
  exit 0
fi

# ── launch ────────────────────────────────────────────────────────────────────
# ROS's setup.bash reads unbound variables (AMENT_TRACE_SETUP_FILES and
# friends), so -u has to come off around the sourcing or it aborts here.
set +u
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
set -u
export IGN_GAZEBO_RESOURCE_PATH=/ros2_ws/install/tb3_rl_bridge/share/tb3_rl_bridge/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}

mkdir -p "$FLEET_DIR"
stop_fleet >/dev/null 2>&1
: > "$FLEET_DIR/pids"

echo "[fleet] launching $N instance(s), stage=$STAGE fast=$FAST domains=${DOMAIN_BASE}..$((DOMAIN_BASE + N - 1))"
echo "[fleet] reward weights: r_vlinear_scale=$R_VLINEAR_SCALE r_vangular_weight=$R_VANGULAR_WEIGHT"

for ((i = 0; i < N; i++)); do
  domain=$((DOMAIN_BASE + i))
  # setsid so each instance gets its own process group and the whole tree can
  # be signalled at once -- ros2 launch otherwise leaves orphaned gz servers.
  setsid env \
    ROS_DOMAIN_ID="$domain" \
    IGN_PARTITION="tb3env${i}" \
    ros2 launch tb3_rl_bridge bridge.launch.py \
      stage:="$STAGE" headless:=true fast:="$FAST" \
      r_vlinear_scale:="$R_VLINEAR_SCALE" \
      r_vangular_weight:="$R_VANGULAR_WEIGHT" \
    > "$FLEET_DIR/env${i}.log" 2>&1 &
  echo $! >> "$FLEET_DIR/pids"
  echo "[fleet]   instance $i -> ROS_DOMAIN_ID=$domain IGN_PARTITION=tb3env${i}"
done

# ── wait for readiness ────────────────────────────────────────────────────────
# A sim that has not finished spawning will accept a /step call and then block,
# so training must not start until every instance advertises its services.
echo "[fleet] waiting for all instances to advertise /step (timeout ${READY_TIMEOUT}s)..."
deadline=$((SECONDS + READY_TIMEOUT))
declare -a ready
for ((i = 0; i < N; i++)); do ready[$i]=0; done

while (( SECONDS < deadline )); do
  all_ready=1
  for ((i = 0; i < N; i++)); do
    if [[ ${ready[$i]} -eq 0 ]]; then
      domain=$((DOMAIN_BASE + i))
      if ROS_DOMAIN_ID=$domain timeout 8 ros2 service list 2>/dev/null | grep -q '^/step$'; then
        ready[$i]=1
        echo "[fleet]   instance $i ready ($((SECONDS))s)"
      else
        all_ready=0
      fi
    fi
  done
  [[ $all_ready -eq 1 ]] && break
  sleep 3
done

n_ready=0
for ((i = 0; i < N; i++)); do (( n_ready += ready[i] )); done

if (( n_ready < N )); then
  echo "[fleet] ERROR only $n_ready/$N instances came up. Check $FLEET_DIR/env*.log" >&2
  echo "[fleet] leaving the fleet running so the logs can be inspected; --stop when done." >&2
  exit 1
fi

echo "[fleet] all $N instances ready. Logs in $FLEET_DIR/env*.log"
echo "[fleet] stop with: scripts/launch_sim_fleet.sh --stop"
