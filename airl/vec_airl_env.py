"""Vectorised AIRL phase-2 env over a fleet of isolated sim instances.

The phase-2 env is the base TurtleBot3 env under three wrappers:

    Strip42to40         drop prev_action dims (the action-echo loop)
    FrameStackFlat(k)   optional temporal context
    FrozenRewardWrapper replace env reward with the recovered reward net

Those classes are imported from airl.train_with_reward rather than
reimplemented here. They define what "the AIRL policy sees" and any
divergence between the single-env and vectorised versions would show up as a
spurious difference in the budget sweep.

The reward net is loaded inside each worker from its path rather than being
pickled across. It carries a RunningNorm layer whose buffers are part of the
frozen artifact; passing a path guarantees every worker deserialises the same
saved state instead of relying on torch pickling to round-trip it.
"""

from __future__ import annotations

import os

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

DEFAULT_DOMAIN_BASE = 40


def _make_airl_env(rank: int, config: dict, domain_base: int,
                   reward_net_path: str, frame_stack: int):
    """Build one worker's phase-2 env. Runs inside the spawned subprocess."""
    os.environ["ROS_DOMAIN_ID"] = str(domain_base + rank)
    os.environ["IGN_PARTITION"] = f"tb3env{rank}"

    # Imported after the domain is set, and inside the worker so that torch
    # does not initialise CUDA in the parent before the fork/spawn boundary.
    from airl.train_with_reward import (
        FrameStackFlat,
        FrozenRewardWrapper,
        Strip42to40,
        load_reward_net,
    )
    from rl.envs.ros2_gym_env import TurtleBot3Env

    env = Strip42to40(TurtleBot3Env(config))
    if frame_stack > 1:
        env = FrameStackFlat(env, k=frame_stack)

    reward_net = load_reward_net(reward_net_path,
                                 env.observation_space,
                                 env.action_space)
    return FrozenRewardWrapper(env, reward_net)


class _AirlEnvFactory:
    """Picklable no-arg factory for SubprocVecEnv under the spawn method."""

    def __init__(self, rank: int, config: dict, domain_base: int,
                 reward_net_path: str, frame_stack: int):
        self.rank = rank
        self.config = config
        self.domain_base = domain_base
        self.reward_net_path = reward_net_path
        self.frame_stack = frame_stack

    def __call__(self):
        return _make_airl_env(self.rank, self.config, self.domain_base,
                              self.reward_net_path, self.frame_stack)


def _make_airl_eval_env(rank: int, config: dict, domain_base: int,
                        frame_stack: int):
    """Build one worker's env for EVALUATION.

    Same observation pipeline as training -- Strip42to40 then optional frame
    stacking, so the policy sees exactly the input it was trained on -- but
    deliberately WITHOUT FrozenRewardWrapper. Evaluation must score the policy
    on the true task: success comes from the environment's own termination and
    reward from its own reward function. Wrapping the learned reward here would
    report how well the policy satisfies its own recovered objective, which is
    not comparable with the other arms.
    """
    os.environ["ROS_DOMAIN_ID"] = str(domain_base + rank)
    os.environ["IGN_PARTITION"] = f"tb3env{rank}"

    from airl.train_with_reward import FrameStackFlat, Strip42to40
    from rl.envs.ros2_gym_env import TurtleBot3Env

    env = Strip42to40(TurtleBot3Env(config))
    if frame_stack > 1:
        env = FrameStackFlat(env, k=frame_stack)
    return env


class _AirlEvalFactory:
    """Picklable no-arg factory for the evaluation env under spawn."""

    def __init__(self, rank: int, config: dict, domain_base: int,
                 frame_stack: int):
        self.rank = rank
        self.config = config
        self.domain_base = domain_base
        self.frame_stack = frame_stack

    def __call__(self):
        return _make_airl_eval_env(self.rank, self.config, self.domain_base,
                                   self.frame_stack)


def make_airl_eval_vec_env(
    config: dict,
    n_envs: int,
    frame_stack: int = 1,
    domain_base: int = DEFAULT_DOMAIN_BASE,
):
    """Vectorised evaluation env for AIRL policies: AIRL's observation space,
    the environment's own reward."""
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    factories = [
        _AirlEvalFactory(i, config, domain_base, frame_stack)
        for i in range(n_envs)
    ]
    return VecMonitor(SubprocVecEnv(factories, start_method="spawn"))


def make_airl_fleet_vec_env(
    config: dict,
    n_envs: int,
    reward_net_path: str,
    frame_stack: int = 1,
    domain_base: int = DEFAULT_DOMAIN_BASE,
):
    """Connect to an already-running fleet of `n_envs` sim instances.

    Does not start the simulators -- launch_sim_fleet.sh does that.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    if not os.path.exists(reward_net_path):
        raise FileNotFoundError(
            f"reward net not found: {reward_net_path}. "
            "Run scripts/prepare_airl.sh to regenerate it from the demos.")

    factories = [
        _AirlEnvFactory(i, config, domain_base, reward_net_path, frame_stack)
        for i in range(n_envs)
    ]
    return VecMonitor(SubprocVecEnv(factories, start_method="spawn"))
