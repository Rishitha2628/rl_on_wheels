#!/usr/bin/env python3
"""Evaluation / rollout entrypoint."""

from __future__ import annotations

import argparse
import os
import sys

import rclpy
import yaml
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(__file__))
from envs.ros2_gym_env import TurtleBot3Env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAC+HER on TurtleBot3")
    p.add_argument("--config",      default="/configs/sac_her.yaml")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to saved model (with or without .zip)")
    p.add_argument("--n-episodes",  type=int, default=None,
                   help="Override eval.n_eval_episodes from config")
    p.add_argument("--stochastic",  action="store_true",
                   help="Use stochastic policy (default: deterministic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not rclpy.ok():
        rclpy.init()

    env = TurtleBot3Env(config)
    model = SAC.load(
        args.checkpoint,
        env=env,
        custom_objects={"tensorboard_log": config["training"]["tensorboard_log"]},
    )

    n_episodes  = args.n_episodes or config["eval"]["n_eval_episodes"]
    deterministic = not args.stochastic and config["eval"]["deterministic"]

    print(f"Evaluating {n_episodes} episodes | deterministic={deterministic}")

    ep_rewards:  list[float] = []
    ep_lengths:  list[int]   = []
    ep_successes: list[bool]  = []

    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        done    = False
        ep_rew  = 0.0
        ep_len  = 0
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done     = terminated or truncated
            ep_rew  += reward
            ep_len  += 1
            if info.get("success"):
                success = True

        ep_rewards.append(ep_rew)
        ep_lengths.append(ep_len)
        ep_successes.append(success)
        print(f"  ep {ep:3d}/{n_episodes}  reward={ep_rew:8.2f}  "
              f"steps={ep_len:4d}  success={success}")

    print("\n── Summary ──")
    print(f"  Mean reward  : {sum(ep_rewards)  / n_episodes:.2f}")
    print(f"  Mean length  : {sum(ep_lengths)  / n_episodes:.1f}")
    print(f"  Success rate : {sum(ep_successes) / n_episodes:.1%}")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
