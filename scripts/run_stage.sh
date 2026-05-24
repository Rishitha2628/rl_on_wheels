#!/usr/bin/env bash
# Run sim + TD3 training for a given drlnav stage.
#
# Usage:
#   ./scripts/run_stage.sh <stage> [checkpoint_path]
#
# Examples:
#   ./scripts/run_stage.sh 1
#   ./scripts/run_stage.sh 2 /checkpoints/td3_tb3_80000_steps
#   ./scripts/run_stage.sh 4 /checkpoints/td3_tb3_180000_steps
#
# Stop with ctrl-C — both sim and trainer come down together.
set -euo pipefail

STAGE="${1:?Usage: $0 <stage 1-10> [checkpoint_path]}"
CHECKPOINT="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "[run_stage] STAGE=$STAGE CHECKPOINT='${CHECKPOINT}'"

STAGE="$STAGE" CHECKPOINT="$CHECKPOINT" \
  docker compose -f docker/docker-compose.yml up \
    --abort-on-container-exit \
    sim train_td3
