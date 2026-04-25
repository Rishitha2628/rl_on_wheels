"""Simple FIFO replay buffer backed by a deque."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Stores (obs, raw_action, reward, done, next_obs) transitions.

    raw_action is in (-1, 1)^2 (tanh space).  The environment action in
    physical units is obtained by the SAC agent via scale_action().
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def add(
        self,
        obs:        np.ndarray,
        raw_action: np.ndarray,
        reward:     float,
        done:       float,      # 0.0 or 1.0 (terminated only, not truncated)
        next_obs:   np.ndarray,
    ) -> None:
        self._buf.append((obs, raw_action, reward, done, next_obs))

    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, dones, next_obs = zip(*batch)
        return (
            np.array(obs,        dtype=np.float32),
            np.array(actions,    dtype=np.float32),
            np.array(rewards,    dtype=np.float32).reshape(-1, 1),
            np.array(dones,      dtype=np.float32).reshape(-1, 1),
            np.array(next_obs,   dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)
