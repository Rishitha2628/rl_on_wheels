#!/usr/bin/env python3
"""Train AIRL on Nav2 expert trajectories.

Pipeline:
  - SB3 PPO is the policy generator.
  - imitation.algorithms.adversarial.airl.AIRL plays the discriminator.
  - Reward net is BasicShapedRewardNet with the AIRL decomposition
        r(s, a, s') = g(s, a) + γ·h(s') - h(s)
    where `g` is the recovered "true" reward and `h` is a state potential
    that absorbs shaping signals — this is what lets the discriminator
    pick out the action-conditional reward separately from the value
    landscape.
  - The wrapped env trims the 42-dim env observation down to the same
    40-dim slice BC uses (drops prev_lin_vel / prev_ang_vel) so the demo
    obs and the env obs match.

Pre-requirements (started separately):
    1. ros2 launch tb3_rl_bridge bridge.launch.py stage:=11 \\
           dynamic_obstacles:=true

Usage (inside container):
    python3 /airl/train_airl.py \\
        --config /configs/airl.yaml \\
        --trajectories /demos/stage11_trajectories.pkl \\
        --total-timesteps 500000 \\
        --out-dir /checkpoints/airl_stage11
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import gymnasium as gym
import numpy as np
import rclpy
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util import logger as imit_logger
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.envs.ros2_gym_env import TurtleBot3Env


class Strip42to40(gym.ObservationWrapper):
    """Drop prev_action dims so AIRL trains on the same 40-dim slice as BC.

    Why: BC's NPZ stores 40-dim obs (we drop prev_lin_vel/prev_ang_vel to
    avoid the action-echo feedback loop). AIRL's demos must match the env's
    obs space, so we slice the env down to the same 40 dims.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = env.observation_space.low[:40]
        high = env.observation_space.high[:40]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:40].astype(np.float32)


def make_env(cfg: dict) -> gym.Env:
    if not rclpy.ok():
        rclpy.init()
    return Strip42to40(TurtleBot3Env(cfg))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="/configs/airl.yaml")
    p.add_argument("--trajectories",    required=True)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--out-dir",         default="/checkpoints/airl_stage11")
    p.add_argument("--seed",            type=int, default=0)
    p.add_argument("--bc-epochs",       type=int, default=20,
                   help="BC pretraining epochs before AIRL. 0 = skip BC warm-start.")
    p.add_argument("--eval-every",      type=int, default=10,
                   help="run env eval every N AIRL rounds (set 0 to disable)")
    p.add_argument("--eval-episodes",   type=int, default=10,
                   help="how many env episodes per eval checkpoint")
    return p.parse_args()


def eval_in_env(policy: PPO, venv, n_episodes: int) -> tuple[float, float]:
    """Run n_episodes deterministic rollouts and return (success_rate, mean_len).

    Used as an AIRL callback so we can snapshot whichever policy is best by
    actual env success rate — not by discriminator reward. AIRL's reward
    metric and env reward are NOT aligned; the policy can keep maximising
    discriminator reward while regressing on actual navigation, as we
    observed in the first 100k-step run.
    """
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


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("/logs/tensorboard/airl", exist_ok=True)

    # ── expert trajectories ─────────────────────────────────────────────────
    with open(args.trajectories, "rb") as f:
        trajectories = pickle.load(f)
    print(f"loaded {len(trajectories)} expert trajectories")

    # ── env ─────────────────────────────────────────────────────────────────
    # Single sim → DummyVecEnv with one env. PPO on-policy will roll out
    # n_steps per update inside this vec env.
    venv = DummyVecEnv([lambda: make_env(cfg)])

    # ── reward net (AIRL decomposition) ─────────────────────────────────────
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # ── PPO generator ───────────────────────────────────────────────────────
    # net_arch matches BC's [512, 512] so the BC pretraining step below can
    # share weight shapes cleanly.
    ppo_cfg = cfg["ppo"]
    policy = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        seed=args.seed,
        n_steps=int(ppo_cfg.get("n_steps", 1024)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        learning_rate=float(ppo_cfg.get("learning_rate", 3.0e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        policy_kwargs=dict(net_arch=[512, 512]),
        tensorboard_log="/logs/tensorboard/airl",
    )

    # ── BC pretraining (warm-start) ─────────────────────────────────────────
    # Without this, PPO starts random and the discriminator immediately wins —
    # we observed the AIRL reward curve descending monotonically because the
    # policy never had enough gradient to catch up. BC pretraining trains the
    # SAME policy object on the demos, so when AIRL takes over, PPO begins
    # already close to Nav2's trajectory distribution and the discriminator
    # can only find subtle gaps (which is what AIRL is good at correcting).
    if args.bc_epochs > 0:
        print(f"BC pretraining for {args.bc_epochs} epochs")
        bc_trainer = BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=rollout.flatten_trajectories(trajectories),
            policy=policy.policy,
            rng=np.random.default_rng(args.seed),
        )
        bc_trainer.train(n_epochs=args.bc_epochs)
        # Tighten the action-distribution noise around the BC mean. PPO's
        # default log_std=0 (std=1) is enormous relative to our action
        # bounds (lin ∈ [0, 0.22], ang ∈ [-2, 2]) — every BC mean would be
        # buried in noise and the discriminator would trivially separate
        # noisy rollouts from clean Nav2 demos. log_std=-1.5 gives std≈0.22
        # which keeps exploration meaningful while staying near BC.
        with torch.no_grad():
            if hasattr(policy.policy, "log_std"):
                policy.policy.log_std.data.fill_(-1.5)
        # Snapshot the BC-only policy so we can A/B against the AIRL-refined one.
        policy.save(os.path.join(args.out_dir, "policy_after_bc"))
        print(f"BC pretraining done; snapshot at {args.out_dir}/policy_after_bc")

    # ── AIRL trainer ────────────────────────────────────────────────────────
    airl_cfg = cfg["airl"]
    airl = AIRL(
        demonstrations=trajectories,
        demo_batch_size=int(airl_cfg.get("demo_batch_size", 2048)),
        gen_replay_buffer_capacity=int(airl_cfg.get("gen_replay_buffer_capacity", 2048)),
        n_disc_updates_per_round=int(airl_cfg.get("n_disc_updates_per_round", 4)),
        venv=venv,
        gen_algo=policy,
        reward_net=reward_net,
        # Our env has variable-length episodes (goal_reached, collision,
        # 450-step timeout). imitation refuses these by default because
        # horizon can leak reward info. For nav tasks where Nav2 demos
        # also have variable horizons, this leak is part of the *task*,
        # not a confound — so we acknowledge it explicitly.
        allow_variable_horizon=True,
        custom_logger=imit_logger.configure(
            folder=os.path.join("/logs/tensorboard/airl", "imit"),
            format_strs=["stdout", "tensorboard"],
        ),
    )

    print(f"training AIRL for {args.total_timesteps} env steps")

    # ── env-eval callback: snapshot policy whenever env success improves ────
    # AIRL's reward (= discriminator score) and env success are NOT aligned —
    # the policy can keep climbing on AIRL reward while regressing on the
    # real task (we observed this on the first run: final policy 16% vs
    # BC-only snapshot 52%). The callback below evaluates the policy in the
    # real env every `eval_every` rounds and saves whichever ckpt has the
    # highest success rate as `policy_best_eval.zip`.
    best_success = -1.0
    best_path = os.path.join(args.out_dir, "policy_best_eval.zip")
    eval_log: list[dict] = []

    def callback(round_num: int) -> None:
        nonlocal best_success
        if args.eval_every <= 0:
            return
        if round_num == 0 or round_num % args.eval_every != 0:
            return
        sr, mean_len = eval_in_env(policy, venv, args.eval_episodes)
        eval_log.append({"round": int(round_num), "success_rate": float(sr),
                         "mean_len": float(mean_len)})
        print(f"[eval @ round {round_num}] success={sr:.2f}  mean_len={mean_len:.0f}")
        if sr > best_success:
            best_success = sr
            policy.save(best_path)
            print(f"  ★ new best — saved {best_path}")

    airl.train(total_timesteps=args.total_timesteps, callback=callback)

    # ── save final policy + reward + eval history ───────────────────────────
    policy.save(os.path.join(args.out_dir, "policy"))
    torch.save(reward_net.state_dict(), os.path.join(args.out_dir, "reward_net.pt"))
    with open(os.path.join(args.out_dir, "eval_log.yaml"), "w") as f:
        yaml.dump({"history": eval_log, "best_success": best_success}, f)
    if best_success >= 0:
        print(f"best env success during AIRL: {best_success:.2f} → "
              f"{best_path}")
    metadata = {
        "obs_dim":    int(venv.observation_space.shape[0]),
        "act_dim":    int(venv.action_space.shape[0]),
        "obs_low":    venv.observation_space.low.tolist(),
        "obs_high":   venv.observation_space.high.tolist(),
        "act_low":    venv.action_space.low.tolist(),
        "act_high":   venv.action_space.high.tolist(),
        "stage":      11,
        "demos":      args.trajectories,
        "timesteps":  args.total_timesteps,
    }
    with open(os.path.join(args.out_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f)
    print(f"saved policy + reward_net + metadata → {args.out_dir}")

    venv.close()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
