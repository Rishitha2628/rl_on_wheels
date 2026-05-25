#!/usr/bin/env python3
"""Training entrypoint for TD3 on TurtleBot3 (frame-stack + action-repeat)."""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import rclpy
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

sys.path.insert(0, os.path.dirname(__file__))

from agents.td3_agent import build, load
from envs.ros2_gym_env import TurtleBot3Env


class ActionRepeat(gym.Wrapper):
    """Hold the agent's action for `n` underlying env steps.

    The policy decides at a slower cadence than the simulator runs, so
    each decision commits the robot to a direction for a meaningful chunk
    of time. Rewards are summed over the repeat window; the final
    transition's done/info is returned.
    """

    def __init__(self, env: gym.Env, n: int):
        super().__init__(env)
        self.n = max(1, int(n))

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs, info = None, {}
        for _ in range(self.n):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TD3 on TurtleBot3")
    p.add_argument("--config",       default="/configs/td3.yaml")
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--reset-buffer", action="store_true",
                   help="On resume, clear the replay buffer (keep weights)")
    return p.parse_args()


def make_vec_env(config: dict):
    """Build the TurtleBot3 env with optional ActionRepeat + VecFrameStack."""
    action_repeat = int(config["td3"].get("action_repeat", 1))
    frame_stack   = int(config["td3"].get("frame_stack",   1))

    def _ctor():
        env = TurtleBot3Env(config)
        if action_repeat > 1:
            env = ActionRepeat(env, action_repeat)
        return env

    vec = DummyVecEnv([_ctor])
    vec = VecMonitor(vec)
    if frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack)
    return vec


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config["training"]["tensorboard_log"], exist_ok=True)

    if not rclpy.ok():
        rclpy.init()

    env = make_vec_env(config)

    if args.checkpoint:
        model = load(args.checkpoint, env, config, reset_buffer=args.reset_buffer)
    else:
        model = build(config, env)

    checkpoint_cb = CheckpointCallback(
        save_freq   = config["training"]["checkpoint_freq"],
        save_path   = checkpoint_dir,
        name_prefix = "td3_tb3",
        verbose     = 1,
    )

    model.learn(
        total_timesteps     = config["td3"]["total_timesteps"],
        callback            = checkpoint_cb,
        log_interval        = config["training"]["log_interval"],
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
