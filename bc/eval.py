#!/usr/bin/env python3
"""Evaluate a trained BC policy in the TurtleBot3 env (deterministic rollout).

Usage:
    python3 /bc/eval.py --config /configs/bc.yaml \
        --checkpoint /checkpoints/bc_best.pt --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rclpy
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bc.networks import BCPolicy
from rl.envs.ros2_gym_env import TurtleBot3Env


def rescale_action(a_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """[-1, 1] → [low, high]."""
    return low + 0.5 * (a_norm + 1.0) * (high - low)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="/configs/bc.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes",   type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = BCPolicy(obs_dim=int(ckpt["obs_dim"]),
                      act_dim=int(ckpt["act_dim"]),
                      hidden=cfg["bc"].get("net_arch", [512, 512])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    act_low  = np.asarray(ckpt["act_low"],  dtype=np.float32)
    act_high = np.asarray(ckpt["act_high"], dtype=np.float32)

    if not rclpy.ok():
        rclpy.init()
    env = TurtleBot3Env(cfg)

    successes = 0
    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len    = 0
        success   = False
        while not done:
            with torch.no_grad():
                o = torch.from_numpy(obs.astype(np.float32))\
                    .unsqueeze(0).to(device)
                a_norm = policy(o).cpu().numpy()[0]
            action = rescale_action(a_norm, act_low, act_high)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(r)
            ep_len += 1
            success = success or bool(info.get("is_success", False))

        successes += int(success)
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        print(f"[ep {ep+1:3d}/{args.episodes}] success={success} "
              f"reward={ep_reward:8.1f} len={ep_len:3d}")

    sr = successes / max(args.episodes, 1)
    print(f"\nResult: success_rate={sr:.2f}  ({successes}/{args.episodes})")
    print(f"  mean reward = {np.mean(ep_rewards):.1f}")
    print(f"  mean length = {np.mean(ep_lengths):.1f}")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
