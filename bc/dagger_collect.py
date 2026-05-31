#!/usr/bin/env python3
"""Run DAgger collection — BC drives the robot while Nav2 silently relabels.

Pre-requirements (started separately in their own terminals):
  1. ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 dynamic_obstacles:=true
  2. ros2 launch tb3_nav2 nav2_bringup.launch.py stage:=4 dagger_mode:=true
     → controller_server publishes velocity commands to /cmd_vel_expert
       (NOT /cmd_vel). The BC policy drives the robot via /cmd_vel, while
       Nav2 produces a "what would I have done here" expert signal at every
       state the BC visits.

At each env step:
  - capture the most-recent /cmd_vel_expert (= Nav2's expert action for the
    current obs)
  - run BC inference → action_bc
  - mix:  with prob β use expert, else use bc  (β defaults to 0 = pure BC)
  - env.step(action_executed)
  - record (obs[:40], expert_action) — same 40-dim layout as the demos NPZ

Output NPZ is compatible with bc/train.py — concatenate it with the existing
demos via dataset_merge.py.

Usage (inside container):
    python3 /bc/dagger_collect.py \
        --config     /configs/bc.yaml \
        --checkpoint /checkpoints/bc_best.pt \
        --out        /demos/dagger_iter1.npz \
        --episodes   200 \
        --beta       0.0
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bc.networks import BCPolicy
from rl.envs.ros2_gym_env import TurtleBot3Env


# ── expert listener ──────────────────────────────────────────────────────────
class ExpertListener(Node):
    """Subscribes to /cmd_vel_expert. Holds the latest message + receive time."""

    def __init__(self) -> None:
        super().__init__("dagger_expert_listener")
        self.lv: float = 0.0
        self.av: float = 0.0
        self.t_last: float = 0.0
        self.create_subscription(Twist, "/cmd_vel_expert", self._on_expert, 10)

    def _on_expert(self, msg: Twist) -> None:
        self.lv = float(msg.linear.x)
        self.av = float(msg.angular.z)
        self.t_last = time.time()

    def fresh(self, max_age: float = 0.3) -> bool:
        return (time.time() - self.t_last) < max_age


def rescale_action(a_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + 0.5 * (a_norm + 1.0) * (high - low)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="/configs/bc.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="path to current BC checkpoint (.pt)")
    p.add_argument("--out",        required=True,
                   help="output NPZ path for new (obs, expert) pairs")
    p.add_argument("--episodes",   type=int,   default=200)
    p.add_argument("--beta",       type=float, default=0.0,
                   help="prob of executing expert instead of BC (default 0 = pure BC drives)")
    p.add_argument("--expert-max-age", type=float, default=0.3,
                   help="discard pairs whose expert signal is older than this (seconds)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── model ────────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCPolicy(obs_dim=int(ckpt["obs_dim"]),
                      act_dim=int(ckpt["act_dim"]),
                      hidden=cfg["bc"].get("net_arch", [512, 512])).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    act_low  = np.asarray(ckpt["act_low"],  dtype=np.float32)
    act_high = np.asarray(ckpt["act_high"], dtype=np.float32)

    # Frame-stack: the policy reads `frame_stack` consecutive 40-dim obs
    # concatenated, but the NPZ we WRITE stores the single 40-dim frame so
    # the merger + next-iter trainer can reapply stacking from ep_id.
    frame_stack  = int(ckpt.get("frame_stack", 1))
    base_obs_dim = int(ckpt["obs_dim"]) // frame_stack

    # ── ROS + env ────────────────────────────────────────────────────────────
    if not rclpy.ok():
        rclpy.init()
    env = TurtleBot3Env(cfg)
    expert = ExpertListener()

    # Spin the expert listener on its own thread so /cmd_vel_expert callbacks
    # fire continuously, regardless of what env.step() is doing on the main
    # thread. A single rclpy.spin_once() per env step is NOT enough — Nav2
    # publishes at ~10 Hz, but env.step() blocks on a service round-trip and
    # the queued messages never get drained between calls.
    expert_exec = SingleThreadedExecutor()
    expert_exec.add_node(expert)
    spin_thread = threading.Thread(target=expert_exec.spin, daemon=True)
    spin_thread.start()

    # Buffers
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_ep:  list[int]        = []

    rng = np.random.default_rng(0)
    n_success = 0
    n_pairs_total  = 0
    n_pairs_stale  = 0

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_pairs_kept = 0
        success = False

        # Give Nav2 a brief moment after the reset to lock onto the new goal
        # before we start asking it for expert actions. The expert spin
        # thread keeps t_last fresh in the background.
        time.sleep(0.25)

        # Seed the frame-stack buffer with the first obs of the episode.
        obs_buf: deque = deque(maxlen=frame_stack)
        for _ in range(frame_stack):
            obs_buf.append(obs[:base_obs_dim].astype(np.float32))

        while not done:
            obs_bc = obs[:base_obs_dim].astype(np.float32)
            obs_buf.append(obs_bc)
            stacked = np.concatenate(list(obs_buf)).astype(np.float32)

            # BC inference
            with torch.no_grad():
                o = torch.from_numpy(stacked).unsqueeze(0).to(device)
                a_norm = policy(o).cpu().numpy()[0]
            action_bc = rescale_action(a_norm, act_low, act_high)

            # Snapshot whatever the background thread has latched.
            expert_lv, expert_av = expert.lv, expert.av
            expert_fresh = expert.fresh(args.expert_max_age)
            n_pairs_total += 1
            if expert_fresh:
                all_obs.append(obs_bc.copy())
                all_act.append(np.array([expert_lv, expert_av], dtype=np.float32))
                all_ep.append(ep - 1)
                ep_pairs_kept += 1
            else:
                n_pairs_stale += 1

            # Beta-mixed execution. β=0 → pure BC drives (standard DAgger-A).
            if args.beta > 0.0 and expert_fresh and rng.random() < args.beta:
                action = np.array([expert_lv, expert_av], dtype=np.float32)
            else:
                action = action_bc

            # ── match eval.py's safety stack exactly ─────────────────────────
            # Train/eval distribution must agree, otherwise BC labels states
            # at deploy time it never saw during DAgger collection (or vice
            # versa). Both the speed cap AND the front-clearance override
            # below are present in eval.py and MUST be present here too.
            action[0] = min(action[0], 0.12)
            front_clearance = obs_bc[39]
            if front_clearance < 0.08:
                action[0] = 0.0
                if abs(action[1]) < 0.5:
                    action[1] = 1.0 if action[1] >= 0 else -1.0

            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = success or bool(info.get("is_success", False))

        n_success += int(success)
        print(f"[ep {ep:4d}/{args.episodes}] success={success}  "
              f"kept_pairs={ep_pairs_kept:3d}  "
              f"total_kept={len(all_obs):6d}  "
              f"({n_success}✓ {ep - n_success}✗)")

    # ── save ────────────────────────────────────────────────────────────────
    if not all_obs:
        print("[warn] no transitions recorded; nothing to save.")
    else:
        out = {
            "obs":    np.stack(all_obs).astype(np.float32),
            "action": np.stack(all_act).astype(np.float32),
            "ep_id":  np.array(all_ep, dtype=np.int32),
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(args.out, **out)
        print(f"\nsaved {len(out['obs'])} (obs, expert) pairs from "
              f"{args.episodes} episodes → {args.out}")
        print(f"  episodes success-rate: {n_success}/{args.episodes} "
              f"({100 * n_success / args.episodes:.1f}%)")
        if n_pairs_total:
            print(f"  dropped stale expert pairs: {n_pairs_stale}/"
                  f"{n_pairs_total} ({100 * n_pairs_stale / n_pairs_total:.1f}%)")

    env.close()
    expert_exec.shutdown()
    expert.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
