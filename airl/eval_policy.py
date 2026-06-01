#!/usr/bin/env python3
"""Evaluate an AIRL/SB3-PPO policy in the live sim (deterministic rollout).

Two checkpoints worth comparing:
  - /checkpoints/airl_stage11/policy_after_bc.zip  → BC-only baseline
  - /checkpoints/airl_stage11/policy.zip           → AIRL-refined

Usage:
    python3 /airl/eval_policy.py \\
        --config /configs/airl.yaml \\
        --policy /checkpoints/airl_stage11/policy.zip \\
        --episodes 50
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import rclpy
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO

from rl.envs.ros2_gym_env import TurtleBot3Env


class Strip42to40(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = env.observation_space.low[:40]
        high = env.observation_space.high[:40]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:40].astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="/configs/airl.yaml")
    p.add_argument("--policy",   required=True, help="path to PPO .zip checkpoint")
    p.add_argument("--episodes", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not rclpy.ok():
        rclpy.init()
    env = Strip42to40(TurtleBot3Env(cfg))

    model = PPO.load(args.policy)
    print(f"loaded policy: {args.policy}")

    n_success = 0
    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(r)
            ep_len += 1
            success = success or bool(info.get("is_success", False))
        n_success += int(success)
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        print(f"[ep {ep:3d}/{args.episodes}] success={success} "
              f"reward={ep_reward:8.1f} len={ep_len:3d}")

    sr = n_success / max(args.episodes, 1)
    print(f"\nResult: success_rate={sr:.2f}  ({n_success}/{args.episodes})")
    print(f"  mean reward = {np.mean(ep_rewards):.1f}")
    print(f"  mean length = {np.mean(ep_lengths):.1f}")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
