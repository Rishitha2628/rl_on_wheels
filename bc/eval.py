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
from collections import deque

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

    # frame_stack: number of consecutive obs the policy expects to be
    # concatenated (oldest..newest). Old checkpoints without this key
    # default to 1 = no stacking.
    frame_stack = int(ckpt.get("frame_stack", 1))
    base_obs_dim = int(ckpt["obs_dim"]) // frame_stack

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
        first_actions: list[np.ndarray] = []  # diagnostic: first 5 actions
        first_obs:     list[np.ndarray] = []
        # Frame-stack ring buffer. Seeded by repeating the initial obs so
        # the first step has a fixed-shape input.
        obs_buf: deque = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            obs_buf.append(obs[:base_obs_dim].astype(np.float32))
        while not done:
            # The env emits a 42-dim obs but BC was trained on the first
            # 40 dims only (we drop prev_lin_vel/prev_ang_vel to avoid the
            # action-echo feedback loop). Slice before feeding.
            obs_bc = obs[:base_obs_dim].astype(np.float32)
            obs_buf.append(obs_bc)
            stacked = np.concatenate(list(obs_buf)).astype(np.float32)
            with torch.no_grad():
                o = torch.from_numpy(stacked).unsqueeze(0).to(device)
                a_norm = policy(o).cpu().numpy()[0]
            action = rescale_action(a_norm, act_low, act_high)
            # Safety speed cap. 0.15 is a compromise — fast enough that BC
            # isn't crippled, slow enough that the wall-scrape override has
            # time to react before contact.
            action[0] = min(action[0], 0.15)

            # Safety / final-approach overrides operate on normalized lidar
            # values, so thresholds scale inversely with max_lidar_range
            # (3.5 m for stage 4, 6.0 m for stage 11).
            max_range = float(cfg["env"].get("max_lidar_range", 3.5))
            front_threshold = 0.45 / max_range
            # Side threshold is tighter (0.25 m) than the head-on threshold
            # (0.45 m). Stage 11's interior corridors regularly put a wall
            # within 0.45 m on one side as the robot threads between walls,
            # which made the previous 0.45 m side guard fire constantly and
            # the robot looked like it stopped mid-path. 0.25 m only fires
            # when the robot is genuinely about to graze the wall.
            side_threshold  = 0.25 / max_range
            front_clearance = obs_bc[39]
            front_left  = obs_bc[:10]
            front_right = obs_bc[27:36]
            min_left  = float(front_left.min())
            min_right = float(front_right.min())

            safety_fired = False

            # Hard safety #1: head-on. Anything in the goal cone within
            # ~0.45 m → kill linear, pivot.
            if front_clearance < front_threshold:
                action[0] = 0.0
                if abs(action[1]) < 0.5:
                    action[1] = 1.0 if action[1] >= 0 else -1.0
                safety_fired = True

            # Hard safety #2: side proximity. Anything in the front 180°
            # within ~0.45 m → kill linear, turn away.
            if min_left < side_threshold or min_right < side_threshold:
                action[0] = 0.0
                if min_left < min_right:
                    action[1] = -1.5
                else:
                    action[1] = +1.5
                safety_fired = True

            # Final-approach override: when no safety guard fired and we're
            # within ~0.7 m of the goal, take over BC's output completely
            # with a simple P-controller: drive forward at 0.10 m/s and
            # steer proportionally to the goal-body angle. BC learned to
            # slow + spin in this region because Nav2's demos all end with
            # a stop at 0.40 m, so it never learned to actually CROSS the
            # success threshold. This handles the last 0.3 m manually.
            dist_to_goal_norm = obs_bc[36]
            near_goal_thresh = 0.7 / max_range
            if not safety_fired and dist_to_goal_norm < near_goal_thresh:
                cos_goal = float(obs_bc[37])
                sin_goal = float(obs_bc[38])
                goal_angle_body = float(np.arctan2(sin_goal, cos_goal))
                action[0] = 0.10
                action[1] = float(np.clip(1.5 * goal_angle_body, -1.0, 1.0))

            if ep_len < 5:
                first_actions.append(action.copy())
                first_obs.append(obs_bc.copy())
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(r)
            ep_len += 1
            success = success or bool(info.get("is_success", False))
        # Diagnostic print for the first episode only
        if ep == 0:
            print("\n── Diagnostic for episode 1 ─────────────────────────")
            print(f"  obs[0][:5]  (first 5 lidar bins, normalised):  {first_obs[0][:5]}")
            print(f"  obs[0][36:] (goal + prev_action):              {first_obs[0][36:]}")
            print(f"  First 5 actions [lin, ang]:")
            for i, a in enumerate(first_actions):
                print(f"     step {i}: lin={a[0]:+.3f}  ang={a[1]:+.3f}")
            print()

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
