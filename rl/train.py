
#!/usr/bin/env python3
"""Training entrypoint for SAC+HER on TurtleBot3."""

from __future__ import annotations

import argparse
import os
import sys

import rclpy
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback

# Make rl/ importable when run directly
sys.path.insert(0, os.path.dirname(__file__))

from agents.sac_her import build, load
from envs.ros2_gym_env import TurtleBot3Env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC+HER on TurtleBot3")
    p.add_argument("--config",     default="/configs/sac_her.yaml",
                   help="Path to sac_her.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Resume from checkpoint path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config["training"]["tensorboard_log"], exist_ok=True)

    if not rclpy.ok():
        rclpy.init()

    env = TurtleBot3Env(config)

    if args.checkpoint:
        model = load(args.checkpoint, env, config)
    else:
        model = build(config, env)

    checkpoint_cb = CheckpointCallback(
        save_freq   = config["training"]["checkpoint_freq"],
        save_path   = checkpoint_dir,
        name_prefix = "sac_her_tb3",
        verbose     = 1,
    )

    model.learn(
        total_timesteps = config["sac"]["total_timesteps"],
        callback        = checkpoint_cb,
        log_interval    = config["training"]["log_interval"],
        reset_num_timesteps = args.checkpoint is None,
    )

    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
