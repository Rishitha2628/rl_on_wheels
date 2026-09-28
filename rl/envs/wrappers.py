"""Env wrappers shared between the single-env and vectorised training paths.

ActionRepeat lived inside rl/train_td3.py. It moved here so the vectorised
path can apply exactly the same wrapper rather than keeping a second copy that
would eventually drift -- the interaction-budget experiment compares runs
across those two paths, so any difference between them is a confound.
"""

from __future__ import annotations

import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    """Hold the agent's action for `n` underlying env steps.

    The policy decides at a slower cadence than the simulator runs, so
    each decision commits the robot to a direction for a meaningful chunk
    of time. Rewards are summed over the repeat window; the final
    transition's done/info is returned.

    Note for budget accounting: this makes one agent step cost `n` bridge
    steps. SB3 counts total_timesteps in wrapped-env steps, so a configured
    1M timesteps at n=4 is 4M actual environment interactions.
    """

    def __init__(self, env: gym.Env, n: int):
        super().__init__(env)
        self.n = max(1, int(n))

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs, info = None, {}
        for _ in range(self.n):
            obs, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
