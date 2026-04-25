#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${CONFIG:-$REPO_ROOT/configs/sac_her.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle_pi}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export PYTHONPATH="$REPO_ROOT/rl:${PYTHONPATH:-}"

source /opt/ros/humble/setup.bash
source "$REPO_ROOT/ros2_ws/install/setup.bash"

echo "[train] Config: $CONFIG"
echo "[train] ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "[train] TensorBoard will be available on port $TENSORBOARD_PORT"

# Launch TensorBoard in background
tensorboard --logdir "$REPO_ROOT/logs/tensorboard" \
            --port "$TENSORBOARD_PORT" \
            --host 0.0.0.0 &
TB_PID=$!
trap "kill $TB_PID 2>/dev/null || true" EXIT

TRAIN_CMD="python3 $REPO_ROOT/rl/train.py --config $CONFIG"
if [[ -n "$CHECKPOINT" ]]; then
  TRAIN_CMD="$TRAIN_CMD --checkpoint $CHECKPOINT"
fi

exec $TRAIN_CMD
