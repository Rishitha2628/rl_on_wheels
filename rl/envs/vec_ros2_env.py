"""Vectorised TurtleBot3 env over a fleet of isolated sim instances.

Pairs with scripts/launch_sim_fleet.sh, which starts N sims on
ROS_DOMAIN_ID = domain_base + i. This module connects one SubprocVecEnv worker
to each of them.

Two things here are load-bearing and easy to get wrong:

**spawn, not fork.** SB3's SubprocVecEnv defaults to fork on Linux. rclpy holds
DDS participants, background threads and file descriptors that do not survive a
fork -- a forked child inherits a context that looks initialised but cannot
communicate. Every worker is therefore spawned, which means the env factory has
to be a picklable module-level callable rather than a lambda or closure.

**ROS_DOMAIN_ID must be set before rclpy.init().** rclpy reads it once when the
context is created; setting it afterwards silently does nothing and all workers
end up on the same domain, fighting over one robot. The assignment happens at
the top of _make_env, before TurtleBot3Env is even imported.
"""

from __future__ import annotations

import os
from typing import Any

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

DEFAULT_DOMAIN_BASE = 40


def _make_env(rank: int, config: dict, domain_base: int):
    """Construct one worker's env. Runs inside the spawned subprocess.

    Module-level and taking only picklable arguments so that spawn can
    reconstruct it on the other side.
    """
    os.environ["ROS_DOMAIN_ID"] = str(domain_base + rank)
    # Matches the partition launch_sim_fleet.sh gave this instance. Only the
    # C++ nodes talk gz-transport, but keeping it set means anything started
    # from this process inherits the right partition rather than the default.
    os.environ["IGN_PARTITION"] = f"tb3env{rank}"

    # Imported here, after the domain is set. (envs/__init__.py pulls rclpy in
    # at package import time regardless, so it is present in the parent too --
    # that is harmless, since only rclpy.init() creates a DDS context and each
    # spawned child builds its own from scratch.)
    from envs.ros2_gym_env import TurtleBot3Env
    from envs.wrappers import ActionRepeat

    env = TurtleBot3Env(config)

    action_repeat = int(config.get("td3", {}).get("action_repeat", 1))
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    return env


class _EnvFactory:
    """Picklable stand-in for `lambda: _make_env(rank, config, base)`.

    functools.partial would also pickle, but SubprocVecEnv calls its factories
    with no arguments, so a small callable class is the clearer fit.
    """

    def __init__(self, rank: int, config: dict, domain_base: int):
        self.rank = rank
        self.config = config
        self.domain_base = domain_base

    def __call__(self):
        return _make_env(self.rank, self.config, self.domain_base)


def make_fleet_vec_env(
    config: dict,
    n_envs: int,
    domain_base: int = DEFAULT_DOMAIN_BASE,
    monitor_dir: str | None = None,
) -> Any:
    """Connect to an already-running fleet of `n_envs` sim instances.

    Does NOT start the simulators -- launch_sim_fleet.sh does that, and it
    blocks until every instance advertises /step. Calling this against a fleet
    that is not up yet will hang on service discovery rather than fail fast.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    factories = [_EnvFactory(i, config, domain_base) for i in range(n_envs)]
    vec = SubprocVecEnv(factories, start_method="spawn")
    return VecMonitor(vec, filename=monitor_dir)
