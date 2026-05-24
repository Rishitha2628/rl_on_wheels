"""SAC agent factory (plain SAC, no HER, using stable-baselines3)."""

from __future__ import annotations

import os

from stable_baselines3 import SAC

from envs.ros2_gym_env import TurtleBot3Env


def build(config: dict, env: TurtleBot3Env) -> SAC:
    """Instantiate a fresh SAC model from config."""
    sac = config["sac"]
    trn = config["training"]

    # SB3 computes log_prob in the pre-rescale tanh-squashed [-1,1] space, so
    # default target_entropy = -action_dim = -2 is the right baseline regardless
    # of the actual action bounds. "auto" lets SB3 set this internally.
    te = sac.get("target_entropy", "auto")
    target_entropy = te if isinstance(te, str) else float(te)

    model = SAC(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = float(sac["learning_rate"]),
        buffer_size            = int(sac["buffer_size"]),
        batch_size             = int(sac["batch_size"]),
        tau                    = float(sac["tau"]),
        gamma                  = float(sac["gamma"]),
        train_freq             = int(sac["train_freq"]),
        gradient_steps         = int(sac["gradient_steps"]),
        ent_coef               = sac.get("ent_coef", "auto"),
        target_entropy         = target_entropy,
        learning_starts        = int(sac["learning_starts"]),
        target_update_interval = int(sac["target_update_interval"]),
        policy_kwargs          = dict(net_arch=[512, 512]),
        tensorboard_log        = trn["tensorboard_log"],
        verbose                = 1,
    )
    return model


def load(checkpoint_path: str, env: TurtleBot3Env, config: dict,
         reset_buffer: bool = False) -> SAC:
    """Resume training from a checkpoint.

    reset_buffer=True drops the old replay buffer so the agent relearns
    Q-values from scratch in the new environment (e.g. after adding obstacles),
    while keeping the learned policy and critic weights.
    """
    if not os.path.exists(checkpoint_path) and not os.path.exists(checkpoint_path + ".zip"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = SAC.load(
        checkpoint_path,
        env=env,
        custom_objects={"tensorboard_log": config["training"]["tensorboard_log"]},
    )
    # custom_objects doesn't reliably override integer hyperparams — set directly
    model.gradient_steps = int(config["sac"]["gradient_steps"])
    if reset_buffer:
        model.replay_buffer.reset()
        print("[load] Replay buffer cleared — starting fresh experience collection.")
    return model
