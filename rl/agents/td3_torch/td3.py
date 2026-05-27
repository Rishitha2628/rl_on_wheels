"""TD3 algorithm — twin critics, delayed policy updates, target smoothing."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck exploration noise (matches SB3's TD3 default)."""

    def __init__(self, size: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.1):
        self.size  = size
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.full(size, mu, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state = self.state + dx
        return self.state.copy()


class TD3:
    """Custom TD3 — actor outputs in [-1, 1], rescaled to env bounds outside."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden: list[int] = (512, 512),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5,
                 device: str = "auto"):
        self.gamma = gamma
        self.tau   = tau
        self.policy_delay        = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip   = target_noise_clip
        self.act_dim             = act_dim

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        hidden = list(hidden)
        self.actor         = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor).to(self.device)
        self.critic        = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Targets never receive gradients.
        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._update_step = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1]. Exploration noise is added outside."""
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        a = self.actor(x).cpu().numpy()[0]
        return a

    def train_step(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, float]:
        if len(buffer) < batch_size:
            return {}

        batch = buffer.sample(batch_size, self.device)
        obs, action, reward, next_obs, done = (
            batch["obs"], batch["action"], batch["reward"],
            batch["next_obs"], batch["done"])

        # ── critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            # target policy smoothing: action picked by actor_target + clipped noise
            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_t = torch.min(q1_t, q2_t)
            target_q = reward + self.gamma * (1.0 - done) * q_t

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info: dict[str, float] = {"critic_loss": float(critic_loss.item())}

        # ── delayed actor + target updates ────────────────────────────────────
        self._update_step += 1
        if self._update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor_target,  self.actor)
            self._soft_update(self.critic_target, self.critic)

            info["actor_loss"] = float(actor_loss.item())

        return info

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(sp.data, alpha=self.tau)

    # ── checkpointing ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":         self.actor.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt":     self.actor_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "update_step":   self._update_step,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self._update_step = int(ckpt.get("update_step", 0))
