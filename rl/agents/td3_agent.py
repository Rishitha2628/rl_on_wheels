"""TD3 agent factory (stable-baselines3) with OU exploration noise."""

from __future__ import annotations

import os

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from envs.ros2_gym_env import TurtleBot3Env


def _make_action_noise(cfg_noise: dict, action_dim: int):
    """Build an action-noise object from config. Returns None if disabled."""
    kind  = (cfg_noise.get("type") or "none").lower()
    sigma = float(cfg_noise.get("sigma", 0.1))
    mean  = np.zeros(action_dim, dtype=np.float32)
    std   = sigma * np.ones(action_dim, dtype=np.float32)

    if kind == "ou":
        theta = float(cfg_noise.get("theta", 0.15))
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=std, theta=theta)
    if kind == "normal" or kind == "gaussian":
        return NormalActionNoise(mean=mean, sigma=std)
    return None


def build(config: dict, env: TurtleBot3Env) -> TD3:
    """Instantiate a fresh TD3 model from config."""
    td3 = config["td3"]
    trn = config["training"]

    action_dim   = env.action_space.shape[0]
    action_noise = _make_action_noise(td3.get("action_noise", {}), action_dim)

    net_arch = td3.get("net_arch", [512, 512])

    model = TD3(
        policy               = "MlpPolicy",
        env                  = env,
        learning_rate        = float(td3["learning_rate"]),
        buffer_size          = int(td3["buffer_size"]),
        batch_size           = int(td3["batch_size"]),
        tau                  = float(td3["tau"]),
        gamma                = float(td3["gamma"]),
        train_freq           = int(td3["train_freq"]),
        gradient_steps       = int(td3["gradient_steps"]),
        learning_starts      = int(td3["learning_starts"]),
        policy_delay         = int(td3.get("policy_delay", 2)),
        target_policy_noise  = float(td3.get("target_policy_noise", 0.2)),
        target_noise_clip    = float(td3.get("target_noise_clip", 0.5)),
        action_noise         = action_noise,
        policy_kwargs        = dict(net_arch=list(net_arch)),
        tensorboard_log      = trn["tensorboard_log"],
        verbose              = 1,
    )
    return model


def load(checkpoint_path: str, env: TurtleBot3Env, config: dict,
         reset_buffer: bool = False) -> TD3:
    """Resume training from a checkpoint.

    reset_buffer=True drops the old replay buffer so the agent relearns
    Q-values from scratch (e.g. after changing the reward / environment),
    while keeping the learned policy and critic weights.
    """
    if not os.path.exists(checkpoint_path) and not os.path.exists(checkpoint_path + ".zip"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = TD3.load(
        checkpoint_path,
        env=env,
        custom_objects={"tensorboard_log": config["training"]["tensorboard_log"]},
    )
    model.gradient_steps = int(config["td3"]["gradient_steps"])
    if reset_buffer:
        model.replay_buffer.reset()
        print("[load] Replay buffer cleared — starting fresh experience collection.")
    return model
