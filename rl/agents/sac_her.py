"""SAC agent factory (plain SAC, no HER, using stable-baselines3)."""

from __future__ import annotations

import os

from stable_baselines3 import SAC

from envs.ros2_gym_env import TurtleBot3Env


def build(config: dict, env: TurtleBot3Env) -> SAC:
    """Instantiate a fresh SAC model from config."""
    sac = config["sac"]
    trn = config["training"]

    # SB3 default target_entropy = -action_dim = -2 is calibrated for symmetric
    # [-1,1] action spaces.  Our [0, max_lv] x [-max_av, max_av] space adds
    # offset = -log(max_lv/2) - log(max_av) ≈ +2.73 to log_prob, which shifts
    # the equilibrium entropy from 2 nats down to 0.73 nats (near-deterministic).
    # Correcting: target_entropy = -(desired_entropy + offset - 2) = -0.73 gives
    # ~2 nats, matching standard SAC behaviour on symmetric spaces.
    target_entropy = float(sac.get("target_entropy", -0.73))

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


def load(checkpoint_path: str, env: TurtleBot3Env, config: dict) -> SAC:
    """Resume training from a checkpoint."""
    if not os.path.exists(checkpoint_path) and not os.path.exists(checkpoint_path + ".zip"):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = SAC.load(
        checkpoint_path,
        env=env,
        custom_objects={"tensorboard_log": config["training"]["tensorboard_log"]},
    )
    return model
