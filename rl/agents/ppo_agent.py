"""PPO agent factory using stable-baselines3."""

from __future__ import annotations

import os

from stable_baselines3 import PPO

from envs.ros2_gym_env import TurtleBot3Env


def build(config: dict, env: TurtleBot3Env) -> PPO:
    ppo = config["ppo"]
    trn = config["training"]

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = float(ppo["learning_rate"]),
        n_steps         = int(ppo["n_steps"]),
        batch_size      = int(ppo["batch_size"]),
        n_epochs        = int(ppo["n_epochs"]),
        gamma           = float(ppo["gamma"]),
        gae_lambda      = float(ppo["gae_lambda"]),
        clip_range      = float(ppo["clip_range"]),
        ent_coef        = float(ppo["ent_coef"]),
        vf_coef         = float(ppo["vf_coef"]),
        max_grad_norm   = float(ppo["max_grad_norm"]),
        policy_kwargs   = dict(net_arch=[512, 512]),
        tensorboard_log = trn["tensorboard_log"],
        verbose         = 1,
    )
    return model


def load(checkpoint_path: str, env: TurtleBot3Env, config: dict) -> PPO:
    if not os.path.exists(checkpoint_path) and not os.path.exists(checkpoint_path + ".zip"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = PPO.load(
        checkpoint_path,
        env=env,
        custom_objects={"tensorboard_log": config["training"]["tensorboard_log"]},
    )
    return model
