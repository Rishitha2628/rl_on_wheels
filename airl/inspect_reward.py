#!/usr/bin/env python3
"""Probe the AIRL-recovered reward over interpretable obs sweeps.

For each sweep we hold a "safe baseline" observation fixed and vary one
field at a time, recording reward_net(s, a, s', done=0). Outputs three
PNG plots and a CSV per sweep.

Sweeps:
  - distance to goal  (obs[36] ∈ [0, 1])      → progress signal
  - min lidar bin     (obs[:36] all at same)  → obstacle proximity
  - goal angle        (sweep cos/sin)          → heading preference

Usage:
    python3 /airl/inspect_reward.py \\
        --checkpoint-dir /checkpoints/airl_stage11 \\
        --out-dir /logs/airl_reward_inspect_stage11
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm


def baseline_obs(obs_dim: int, frame_stack: int = 1) -> np.ndarray:
    """Safe baseline obs. For frame-stacked input (obs_dim = frame_stack * 40)
    we set every stacked frame to the same safe values so a single sweep
    dim is interpretable in spite of the stacking.

    - All lidar bins at 1.0 (no obstacle within range).
    - Mid-distance to goal (0.5 normalized).
    - Goal directly ahead (cos=1, sin=0).
    - Goal path clear (obs[39] = 1.0).
    """
    base_dim = obs_dim // frame_stack
    single = np.ones(base_dim, dtype=np.float32)
    single[36] = 0.5
    single[37] = 1.0
    single[38] = 0.0
    single[39] = 1.0
    return np.tile(single, frame_stack).astype(np.float32)


def _idx(field: int, frame_stack: int, base_dim: int = 40) -> int:
    """For a frame-stacked obs, point at `field` in the LATEST frame (rightmost block)."""
    return base_dim * (frame_stack - 1) + field


def query_reward(reward_net, obs: np.ndarray, action: np.ndarray,
                 next_obs: np.ndarray) -> float:
    with torch.no_grad():
        r = reward_net(
            torch.from_numpy(obs).unsqueeze(0),
            torch.from_numpy(action).unsqueeze(0),
            torch.from_numpy(next_obs).unsqueeze(0),
            torch.zeros(1),
        ).item()
    return float(r)


def sweep_distance(reward_net, obs_dim: int, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Vary dist-to-goal in the LATEST stacked frame."""
    dists = np.linspace(0.0, 1.0, 50)
    rewards = np.zeros_like(dists)
    base = baseline_obs(obs_dim, fs)
    action = np.array([0.15, 0.0], dtype=np.float32)
    i36 = _idx(36, fs)
    for i, d in enumerate(dists):
        s = base.copy(); s[i36] = d
        s_next = s.copy(); s_next[i36] = max(0.0, d - 0.02)
        rewards[i] = query_reward(reward_net, s, action, s_next)
    return dists, rewards


def sweep_lidar(reward_net, obs_dim: int, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Vary uniform lidar value across ALL 36 bins of the latest frame."""
    lidars = np.linspace(0.02, 1.0, 50)
    rewards = np.zeros_like(lidars)
    base = baseline_obs(obs_dim, fs)
    action = np.array([0.15, 0.0], dtype=np.float32)
    base_off = 40 * (fs - 1)
    i39 = _idx(39, fs)
    for i, l in enumerate(lidars):
        s = base.copy()
        s[base_off:base_off + 36] = l   # all lidar bins of latest frame
        s[i39] = l
        s_next = s.copy()
        rewards[i] = query_reward(reward_net, s, action, s_next)
    return lidars, rewards


def sweep_goal_angle(reward_net, obs_dim: int, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Vary goal heading in the latest frame."""
    angles = np.linspace(-np.pi, np.pi, 60)
    rewards = np.zeros_like(angles)
    base = baseline_obs(obs_dim, fs)
    action = np.array([0.15, 0.0], dtype=np.float32)
    i37 = _idx(37, fs)
    i38 = _idx(38, fs)
    for i, a in enumerate(angles):
        s = base.copy()
        s[i37] = float(np.cos(a))
        s[i38] = float(np.sin(a))
        s_next = s.copy()
        rewards[i] = query_reward(reward_net, s, action, s_next)
    return angles, rewards


def save_plot_and_csv(x: np.ndarray, y: np.ndarray, xlabel: str, title: str,
                      out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AIRL reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=120)
    plt.close(fig)
    np.savetxt(out_dir / f"{stem}.csv",
               np.column_stack([x, y]),
               delimiter=",",
               header=f"{xlabel},reward",
               comments="")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                   help="directory containing reward_net.pt + metadata.yaml")
    p.add_argument("--out-dir", default="/logs/airl_reward_inspect")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    obs_dim = int(meta["obs_dim"])
    obs_low  = np.array(meta["obs_low"],  dtype=np.float32)
    obs_high = np.array(meta["obs_high"], dtype=np.float32)
    act_low  = np.array(meta["act_low"],  dtype=np.float32)
    act_high = np.array(meta["act_high"], dtype=np.float32)

    obs_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
    act_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    # Must match train_airl.py's construction (incl. RunningNorm) so the
    # saved state_dict keys line up exactly.
    reward_net = BasicShapedRewardNet(
        observation_space=obs_space,
        action_space=act_space,
        normalize_input_layer=RunningNorm,
    )
    reward_net.load_state_dict(torch.load(ckpt_dir / "reward_net.pt",
                                          map_location="cpu"))
    reward_net.eval()

    # Detect frame stacking from obs_dim (40 base dims per stacked frame).
    fs = max(1, obs_dim // 40)
    if fs > 1:
        print(f"detected frame_stack={fs} (obs_dim={obs_dim}) — sweeping the latest frame")

    dx, dy = sweep_distance(reward_net, obs_dim, fs)
    save_plot_and_csv(dx, dy, "dist_to_goal (normalized)",
                      "Reward vs distance to goal", out_dir,
                      "reward_vs_distance")
    print(f"  distance sweep: reward range [{dy.min():.3f}, {dy.max():.3f}]")

    lx, ly = sweep_lidar(reward_net, obs_dim, fs)
    save_plot_and_csv(lx, ly, "min lidar clearance (normalized)",
                      "Reward vs obstacle proximity", out_dir,
                      "reward_vs_lidar")
    print(f"  lidar sweep:    reward range [{ly.min():.3f}, {ly.max():.3f}]")

    ax_, ay = sweep_goal_angle(reward_net, obs_dim, fs)
    save_plot_and_csv(ax_, ay, "goal angle (rad, body frame)",
                      "Reward vs goal heading", out_dir,
                      "reward_vs_goal_angle")
    print(f"  angle sweep:    reward range [{ay.min():.3f}, {ay.max():.3f}]")

    print(f"\nsaved 3 plots + 3 CSVs → {out_dir}")


if __name__ == "__main__":
    main()
