"""
Evaluation-time substrate interventions.

These wrappers degrade the layers *beneath* a trained policy without touching
the policy itself: the sensor it reads and the actuator it drives. A fixed
checkpoint is replayed against a changed substrate, so what they measure is
the robustness of an already-trained policy, not whether training under these
conditions would help.

Both are applied at eval time only, which is the point: they are exactly the
kind of hidden component that an in-distribution evaluation never varies.
"""
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym


class LidarDegradation(gym.Wrapper):
    """
    Gaussian range noise, random dropouts, and a shortened maximum range.

    The bridge hands over ``lidar_bins`` sector minima already normalised by
    ``max_lidar_range`` (``bins[i] = min_val / rmax`` in env_bridge_node.cpp),
    so noise is specified in METRES here and converted internally.

    ``obs[bins + 3]`` (goal_path_min_lidar) is also lidar-derived. It is
    recomputed from the degraded bins using the same ±2-bin cone the C++ uses,
    so the policy cannot read an undegraded copy of the scan through it.

    A dropout and an out-of-range return both read as max range, matching the
    bridge, which initialises ``lidar_data_`` to 1.0 (max-range) and clamps
    beyond ``rmax``.
    """

    GOAL_HALF_WINDOW = 2  # ±2 bins = ±20° at 36 bins, as in env_bridge_node.cpp

    def __init__(self, env, *, noise_m: float = 0.0, dropout_p: float = 0.0,
                 max_range_m: float | None = None, rmax: float = 6.0,
                 bins: int = 36, seed: int = 0) -> None:
        super().__init__(env)
        self.noise_m = float(noise_m)
        self.dropout_p = float(dropout_p)
        self.max_range_m = None if max_range_m is None else float(max_range_m)
        self.rmax = float(rmax)
        self.bins = int(bins)
        self._rng = np.random.default_rng(seed)

    def _degrade(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).copy()
        b = self.bins
        d = o[:b].astype(np.float64) * self.rmax          # → metres

        # A sensor that cannot see past max_range_m reports no return, which
        # the bridge represents as max range.
        if self.max_range_m is not None:
            d = np.where(d > self.max_range_m, self.rmax, d)
        if self.noise_m > 0.0:
            d = d + self._rng.normal(0.0, self.noise_m, size=b)
        if self.dropout_p > 0.0:
            d = np.where(self._rng.random(b) < self.dropout_p, self.rmax, d)

        d = np.clip(d, 0.0, self.rmax)
        o[:b] = (d / self.rmax).astype(np.float32)

        # Recompute goal_path_min_lidar from the degraded bins. int() truncation
        # and the fmod wrap mirror the C++ exactly.
        cos_g, sin_g = float(o[b + 1]), float(o[b + 2])
        ang = math.atan2(sin_g, cos_g)
        pos = math.fmod(ang + 2.0 * math.pi, 2.0 * math.pi)
        center = int(pos / (2.0 * math.pi) * b) % b
        w = self.GOAL_HALF_WINDOW
        o[b + 3] = min(float(o[(center + k) % b]) for k in range(-w, w + 1))
        return o

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._degrade(obs), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._degrade(obs), r, terminated, truncated, info


class ActuationLag(gym.Wrapper):
    """
    First-order actuator lag: the wheels do not reach the commanded velocity
    instantly but approach it with time constant ``tau_sec``.

        executed += (1 - exp(-dt / tau)) * (command - executed)

    ``dt`` is one bridge step (``step_duration_sec``, 0.1 s), which is the rate
    this wrapper is stepped at even when the caller applies an action repeat.
    tau = 0 reduces to the identity.
    """

    def __init__(self, env, *, tau_sec: float = 0.0, dt: float = 0.1) -> None:
        super().__init__(env)
        self.tau_sec = float(tau_sec)
        self.dt = float(dt)
        self._alpha = 1.0 if self.tau_sec <= 0.0 else 1.0 - math.exp(-self.dt / self.tau_sec)
        self._executed = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        self._executed = np.zeros(2, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        cmd = np.asarray(action, dtype=np.float32)
        self._executed = self._executed + self._alpha * (cmd - self._executed)
        return self.env.step(self._executed.copy())
