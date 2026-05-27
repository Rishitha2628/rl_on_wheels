"""Behavior cloning network — MLP policy that maps obs → action."""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class BCPolicy(nn.Module):
    """Deterministic policy: obs → action in [-1, 1] (rescaled at eval time)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = list(hidden) if hidden else [512, 512]
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))
