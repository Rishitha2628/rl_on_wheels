"""Actor and twin Critic networks for TD3."""

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


class Actor(nn.Module):
    """Outputs actions in [-1, 1]; the agent rescales to env bounds."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Twin Q networks (Q1, Q2) — both inputs are (obs, action) concatenated."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))
