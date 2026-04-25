#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Ignition Fortress resource paths (replaces GAZEBO_MODEL_PATH / GAZEBO_PLUGIN_PATH)
export IGN_GAZEBO_RESOURCE_PATH=/opt/ros/humble/share/tb3_rl_bridge/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:-}

exec "$@"
