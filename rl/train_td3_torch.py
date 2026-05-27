#!/usr/bin/env python3
"""Custom PyTorch TD3 training entrypoint for TurtleBot3 (curriculum branch).

Differences from rl/train_td3.py (SB3-based):
    - Hand-written TD3 algo in rl/agents/td3_torch/
    - Frame stacking is implemented inline (stacked observations are
      concatenated and fed to the actor/critic each step)
    - No VecMonitor / VecFrameStack — everything in this file
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))

from agents.td3_torch import TD3
from agents.td3_torch.replay_buffer import ReplayBuffer
from agents.td3_torch.td3 import OUNoise
from envs.ros2_gym_env import TurtleBot3Env


# ── helpers ───────────────────────────────────────────────────────────────────
def rescale_action(a_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """[-1, 1] → [low, high] (linear)."""
    return low + 0.5 * (a_norm + 1.0) * (high - low)


def stack_obs(buf: deque) -> np.ndarray:
    """Concatenate the deque of frames into a single flat observation."""
    return np.concatenate(list(buf), axis=0).astype(np.float32)


def fill_buf(buf: deque, frame: np.ndarray, n: int) -> None:
    buf.clear()
    for _ in range(n):
        buf.append(frame)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="/configs/td3_torch.yaml")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    td3_cfg  = cfg["td3"]
    trn_cfg  = cfg["training"]

    frame_stack   = int(td3_cfg.get("frame_stack",   1))
    action_repeat = int(td3_cfg.get("action_repeat", 1))

    os.makedirs(trn_cfg["checkpoint_dir"],  exist_ok=True)
    os.makedirs(trn_cfg["tensorboard_log"], exist_ok=True)

    if not rclpy.ok():
        rclpy.init()
    env = TurtleBot3Env(cfg)

    base_obs_dim = int(np.prod(env.observation_space.shape))
    obs_dim      = base_obs_dim * frame_stack
    act_dim      = int(env.action_space.shape[0])
    act_low      = env.action_space.low.astype(np.float32)
    act_high     = env.action_space.high.astype(np.float32)

    agent = TD3(
        obs_dim             = obs_dim,
        act_dim             = act_dim,
        hidden              = td3_cfg.get("net_arch", [512, 512]),
        actor_lr            = float(td3_cfg["learning_rate"]),
        critic_lr           = float(td3_cfg["learning_rate"]),
        gamma               = float(td3_cfg["gamma"]),
        tau                 = float(td3_cfg["tau"]),
        policy_delay        = int(td3_cfg["policy_delay"]),
        target_policy_noise = float(td3_cfg["target_policy_noise"]),
        target_noise_clip   = float(td3_cfg["target_noise_clip"]),
    )

    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"[load] resumed from {args.checkpoint}")

    buffer = ReplayBuffer(
        capacity = int(td3_cfg["buffer_size"]),
        obs_dim  = obs_dim,
        act_dim  = act_dim,
    )

    noise_cfg = td3_cfg.get("action_noise", {}) or {}
    noise = OUNoise(
        size  = act_dim,
        theta = float(noise_cfg.get("theta", 0.15)),
        sigma = float(noise_cfg.get("sigma", 0.1)),
    )

    writer = SummaryWriter(log_dir=trn_cfg["tensorboard_log"])

    total_timesteps   = int(td3_cfg["total_timesteps"])
    learning_starts   = int(td3_cfg["learning_starts"])
    batch_size        = int(td3_cfg["batch_size"])
    gradient_steps    = int(td3_cfg.get("gradient_steps", 1))
    checkpoint_freq   = int(trn_cfg["checkpoint_freq"])
    log_interval      = int(trn_cfg.get("log_interval", 10))

    # ── rollout loop ──────────────────────────────────────────────────────────
    frames: deque = deque(maxlen=frame_stack)
    raw_obs, _ = env.reset()
    fill_buf(frames, raw_obs, frame_stack)
    obs = stack_obs(frames)

    noise.reset()
    ep_reward = 0.0
    ep_len    = 0
    ep_count  = 0
    successes: deque = deque(maxlen=100)  # rolling success rate over 100 eps
    t_start = time.time()

    for step in range(1, total_timesteps + 1):
        # action selection
        if step < learning_starts:
            a_norm = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            a_norm = agent.select_action(obs) + noise.sample()
            a_norm = np.clip(a_norm, -1.0, 1.0)
        action = rescale_action(a_norm, act_low, act_high)

        # action repeat — hold the action for several env steps
        total_r = 0.0
        terminated = truncated = False
        info: dict = {}
        for _ in range(action_repeat):
            raw_next, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            if terminated or truncated:
                break

        frames.append(raw_next)
        next_obs = stack_obs(frames)
        done = terminated or truncated
        buffer.add(obs, a_norm, total_r, next_obs, terminated)
        obs = next_obs

        ep_reward += total_r
        ep_len    += 1

        # gradient updates
        if step >= learning_starts:
            for _ in range(gradient_steps):
                metrics = agent.train_step(buffer, batch_size)
            if metrics:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)

        # episode boundary
        if done:
            ep_count += 1
            success = bool(info.get("is_success", False))
            successes.append(success)

            writer.add_scalar("rollout/ep_reward",   ep_reward, step)
            writer.add_scalar("rollout/ep_len",      ep_len,    step)
            writer.add_scalar("rollout/success_rate",
                              float(np.mean(successes)) if successes else 0.0, step)

            if ep_count % log_interval == 0:
                fps = step / max(time.time() - t_start, 1e-9)
                print(f"[ep {ep_count:5d} step {step:7d}] "
                      f"reward={ep_reward:8.1f} len={ep_len:3d} "
                      f"success={np.mean(successes):.2f} fps={fps:.1f}")

            raw_obs, _ = env.reset()
            fill_buf(frames, raw_obs, frame_stack)
            obs = stack_obs(frames)
            noise.reset()
            ep_reward = 0.0
            ep_len    = 0

        # checkpointing
        if step % checkpoint_freq == 0:
            ckpt = os.path.join(trn_cfg["checkpoint_dir"],
                                f"td3_torch_{step}_steps.pt")
            agent.save(ckpt)
            print(f"[ckpt] saved {ckpt}")

    final = os.path.join(trn_cfg["checkpoint_dir"], "td3_torch_final.pt")
    agent.save(final)
    print(f"Training complete. Final checkpoint: {final}")

    env.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
