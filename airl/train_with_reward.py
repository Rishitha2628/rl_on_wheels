#!/usr/bin/env python3
"""Phase 2 of IRL: train PPO against the frozen, AIRL-recovered reward.

Why this exists: pure AIRL (the adversarial loop in train_airl.py) made
the policy worse than its BC warm-start. The discriminator's reward is a
moving target during AIRL training, and PPO ended up chasing adversarial
shortcuts that don't translate to env success.

This script does the cleaner thing: take the reward function AIRL already
recovered, FREEZE it, and run plain PPO against it. Stationary reward →
stable convergence → policy can actually exceed BC if the recovered
reward is good.

Inputs:
  - /checkpoints/airl_stage11_v2/reward_net.pt        frozen reward
  - /checkpoints/airl_stage11_v2/policy_after_bc.zip  BC warm-start

Pre-requirements (in another terminal):
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=11 dynamic_obstacles:=true

Usage:
    python3 /airl/train_with_reward.py \\
        --reward-net /checkpoints/airl_stage11_v2/reward_net.pt \\
        --init-policy /checkpoints/airl_stage11_v2/policy_after_bc.zip \\
        --config /configs/airl.yaml \\
        --total-timesteps 200000 \\
        --out-dir /checkpoints/airl_phase2_stage11
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.envs.ros2_gym_env import TurtleBot3Env


class Strip42to40(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = env.observation_space.low[:40]
        high = env.observation_space.high[:40]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:40].astype(np.float32)


class FrozenRewardWrapper(gym.Wrapper):
    """Replace env reward with reward_net(s, a, s').

    The reward net is frozen (eval mode, no grad). PPO sees a stationary
    reward landscape — no more adversarial chase, no more drift.
    """

    def __init__(self, env: gym.Env, reward_net: torch.nn.Module) -> None:
        super().__init__(env)
        self.reward_net = reward_net.eval()
        for p in self.reward_net.parameters():
            p.requires_grad_(False)
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _env_r, terminated, truncated, info = self.env.step(action)
        assert self._last_obs is not None
        with torch.no_grad():
            r_t = self.reward_net(
                torch.from_numpy(self._last_obs).unsqueeze(0).float(),
                torch.from_numpy(np.asarray(action, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(obs).unsqueeze(0).float(),
                torch.zeros(1),
            )
        r = float(r_t.item())
        self._last_obs = obs.copy()
        return obs, r, terminated, truncated, info


def load_reward_net(reward_net_path: str, obs_space: gym.spaces.Box,
                    act_space: gym.spaces.Box) -> BasicShapedRewardNet:
    net = BasicShapedRewardNet(
        observation_space=obs_space,
        action_space=act_space,
        normalize_input_layer=RunningNorm,
    )
    net.load_state_dict(torch.load(reward_net_path, map_location="cpu"))
    return net


def eval_in_env(policy: PPO, venv, n_episodes: int) -> tuple[float, float]:
    """Deterministic env evaluation — returns (success_rate, mean_len)."""
    n_success = 0
    ep_lens: list[int] = []
    for _ in range(n_episodes):
        obs = venv.reset()
        done = False
        steps = 0
        success = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if info.get("is_success", False):
                success = True
            steps += 1
        n_success += int(success)
        ep_lens.append(steps)
    return n_success / max(n_episodes, 1), float(np.mean(ep_lens))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reward-net",     required=True)
    p.add_argument("--init-policy",    required=True,
                   help="BC warm-start PPO checkpoint (policy_after_bc.zip)")
    p.add_argument("--config",         default="/configs/airl.yaml")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--out-dir",        default="/checkpoints/airl_phase2_stage11")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--eval-every",     type=int, default=10_000,
                   help="run env eval every N env steps")
    p.add_argument("--eval-episodes",  type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── env (reward replaced by frozen reward net) ──────────────────────────
    if not rclpy.ok():
        rclpy.init()

    base_env = Strip42to40(TurtleBot3Env(cfg))
    reward_net = load_reward_net(args.reward_net,
                                 base_env.observation_space,
                                 base_env.action_space)
    wrapped = FrozenRewardWrapper(base_env, reward_net)
    venv = DummyVecEnv([lambda: wrapped])

    # ── load BC warm-start policy ───────────────────────────────────────────
    print(f"loading BC warm-start: {args.init_policy}")
    policy = PPO.load(
        args.init_policy,
        env=venv,
        # Re-use the same hyperparameters the original PPO had.
        # The .zip already encodes the policy/value net weights + sizes.
        tensorboard_log="/logs/tensorboard/airl_phase2",
        verbose=1,
        seed=args.seed,
    )

    # ── training loop with periodic env-eval-based checkpointing ────────────
    # We call policy.learn() in chunks so we can eval between chunks. Each
    # chunk = `eval_every` env steps. After every chunk we evaluate in the
    # TRUE env (not the frozen reward proxy) and snapshot the best policy.
    best_success = -1.0
    best_path = os.path.join(args.out_dir, "policy_best_eval.zip")
    eval_log: list[dict] = []

    total = args.total_timesteps
    chunk = args.eval_every
    steps_done = 0

    print(f"phase 2: training PPO against frozen AIRL reward for {total} steps")
    while steps_done < total:
        n = min(chunk, total - steps_done)
        policy.learn(total_timesteps=n, reset_num_timesteps=False,
                     progress_bar=False)
        steps_done += n

        sr, mean_len = eval_in_env(policy, venv, args.eval_episodes)
        eval_log.append({"steps": int(steps_done),
                         "success_rate": float(sr),
                         "mean_len": float(mean_len)})
        print(f"[eval @ {steps_done} steps] success={sr:.2f}  mean_len={mean_len:.0f}")
        if sr > best_success:
            best_success = sr
            policy.save(best_path)
            print(f"  ★ new best — saved {best_path}")

    # ── final save ──────────────────────────────────────────────────────────
    policy.save(os.path.join(args.out_dir, "policy"))
    with open(os.path.join(args.out_dir, "eval_log.yaml"), "w") as f:
        yaml.dump({"history": eval_log, "best_success": best_success}, f)
    print(f"\ndone. best env success = {best_success:.2f}")
    print(f"  best   → {best_path}")
    print(f"  final  → {args.out_dir}/policy.zip")

    venv.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
