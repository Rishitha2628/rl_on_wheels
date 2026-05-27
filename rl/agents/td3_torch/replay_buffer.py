"""Simple FIFO replay buffer for off-policy RL."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Stores transitions (obs, action, reward, next_obs, done) in numpy arrays.

    All shapes are pre-allocated for speed. `sample(batch_size)` returns torch
    tensors on the requested device.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action   = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward   = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i]      = obs
        self.action[i]   = action
        self.reward[i]   = reward
        self.next_obs[i] = next_obs
        self.done[i]     = float(done)
        self.ptr  = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).to(device),
            "action":   torch.from_numpy(self.action[idx]).to(device),
            "reward":   torch.from_numpy(self.reward[idx]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(device),
            "done":     torch.from_numpy(self.done[idx]).to(device),
        }

    def reset(self) -> None:
        self.ptr  = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size
