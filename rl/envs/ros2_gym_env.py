"""
TurtleBot3 Gym environment backed by ROS2 bridge services.

Observation space:
  Box(41,) — 36 min-range lidar bins  [0, 1]
            + dist_to_goal            [0, 1]  (normalised by max_lidar_range)
            + cos(goal_angle)         [-1, 1] (goal direction in robot frame)
            + sin(goal_angle)         [-1, 1]
            + prev_lin_vel            [-max_lv, max_lv]
            + prev_ang_vel            [-max_av, max_av]

Action space:
  Box(2,) — [linear_vel (m/s), angular_vel (rad/s)]
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.node import Node

try:
    from tb3_rl_bridge.srv import GetObservation, ResetEpisode, Step
except ImportError as exc:
    raise ImportError(
        "tb3_rl_bridge ROS2 package not found. "
        "Build the workspace: colcon build inside ros2_ws/"
    ) from exc


class _BridgeClient(Node):
    """Thin rclpy node that exposes synchronous wrappers for bridge services."""

    def __init__(self, cfg_ros2: dict) -> None:
        super().__init__("tb3_gym_client")
        timeout = cfg_ros2.get("service_timeout_sec", 10.0)

        self._obs_client   = self.create_client(GetObservation, cfg_ros2["observation_service"])
        self._step_client  = self.create_client(Step,           cfg_ros2["step_service"])
        self._reset_client = self.create_client(ResetEpisode,   cfg_ros2["reset_service"])
        self._timeout      = float(timeout)

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=self._timeout):
            raise RuntimeError(f"ROS2 service not available: {client.srv_name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)
        result = future.result()
        if result is None:
            raise RuntimeError(f"ROS2 service call timed out: {client.srv_name}")
        return result

    def get_observation(self):
        return self.call(self._obs_client, GetObservation.Request())

    def step(self, linear_vel: float, angular_vel: float):
        req = Step.Request()
        req.action = [linear_vel, angular_vel]
        return self.call(self._step_client, req)

    def reset_episode(self, *, random_pose: bool = True,
                      spawn_x: float = 0.0, spawn_y: float = 0.0,
                      spawn_theta: float = 0.0,
                      goal_x: float = 1.0, goal_y: float = 0.0):
        req = ResetEpisode.Request()
        req.random_pose  = random_pose
        req.spawn_x      = spawn_x
        req.spawn_y      = spawn_y
        req.spawn_theta  = spawn_theta
        req.goal_x       = goal_x
        req.goal_y       = goal_y
        return self.call(self._reset_client, req)


class TurtleBot3Env(gym.Env):
    """
    gymnasium.Env wrapper around the ROS2 tb3_rl_bridge.

    Flat Box observation — goal-relative features are pre-computed in C++
    so the policy receives directly actionable input without needing to
    subtract or rotate goal coordinates internally.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg_env  = config["env"]
        cfg_ros2 = config["ros2"]

        self._lidar_bins      = int(cfg_env["lidar_bins"])       # 36
        self._max_steps       = int(cfg_env["max_episode_steps"])
        self._goal_tolerance  = float(cfg_env["goal_tolerance"])
        self._max_lidar_range = float(cfg_env.get("max_lidar_range", 3.5))

        max_lv = float(cfg_env["max_linear_vel"])   # 0.26 m/s
        max_av = float(cfg_env["max_angular_vel"])  # 0.4 rad/s

        # ── observation: [min_lidar(36), dist_norm, cos_goal, sin_goal, prev_lv, prev_av]
        obs_low  = np.concatenate([
            np.zeros(self._lidar_bins),
            np.array([0.0, -1.0, -1.0, 0.0, -max_av]),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            np.ones(self._lidar_bins),
            np.array([1.0,  1.0,  1.0,  max_lv,  max_av]),
        ]).astype(np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low  = np.array([0.0,    -max_av], dtype=np.float32),
            high = np.array([max_lv,  max_av], dtype=np.float32),
            dtype = np.float32,
        )

        # ── ROS2 client ───────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()
        self._client = _BridgeClient(cfg_ros2)

        self._step_count = 0

    # ── gym.Env interface ─────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        safe_norm = (self._goal_tolerance * 2) / self._max_lidar_range
        obs = None
        for _ in range(10):
            reset_res = self._client.reset_episode(random_pose=True)
            time.sleep(0.2)
            if not reset_res.success:
                time.sleep(0.5)
                continue
            res = self._client.get_observation()
            obs = self._parse(res)
            if obs[:self._lidar_bins].min() > safe_norm:
                break

        if obs is None:
            res = self._client.get_observation()
            obs = self._parse(res)

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        lv = float(np.clip(action[0],
                           self.action_space.low[0],
                           self.action_space.high[0]))
        av = float(np.clip(action[1],
                           self.action_space.low[1],
                           self.action_space.high[1]))

        res = self._client.step(lv, av)

        obs        = self._parse(res)
        # Scale rewards to match the reference regime (terminal ±100, step ~±1).
        # env_bridge_node emits terminal ±1.0 and step rewards in [-0.01, 0.01];
        # multiplying by 100 keeps the same ratios while giving auto-temperature
        # a reward signal large enough to prevent immediate alpha collapse.
        reward     = float(res.reward) * 100.0
        terminated = bool(res.done)
        truncated  = self._step_count >= self._max_steps
        info: dict[str, Any] = {
            "is_success": res.info == "goal_reached",
            "info":       res.info,
        }
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.destroy_node()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse(res) -> np.ndarray:
        return np.array(res.observation, dtype=np.float32)
