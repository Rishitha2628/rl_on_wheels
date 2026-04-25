#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export DISPLAY="${DISPLAY:-:0}"

HEADLESS="${HEADLESS:-0}"

if [[ "$HEADLESS" == "1" ]]; then
  echo "[launch_sim] Starting in headless mode (Ignition server-only)"
  LAUNCH_ARGS="headless:=true"
else
  LAUNCH_ARGS="headless:=false"
fi

echo "[launch_sim] TurtleBot3 model: $TURTLEBOT3_MODEL"
echo "[launch_sim] ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2_ws/install/setup.bash"

exec ros2 launch tb3_rl_bridge bridge.launch.py $LAUNCH_ARGS
