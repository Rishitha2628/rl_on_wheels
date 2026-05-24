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
                      goal_x: float = 1.0, goal_y: float = 0.0,
                      spawn_only: bool = False,
                      goal_radius: float = 1.0):
        req = ResetEpisode.Request()
        req.random_pose  = random_pose
        req.spawn_x      = spawn_x
        req.spawn_y      = spawn_y
        req.spawn_theta  = spawn_theta
        req.goal_x       = goal_x
        req.goal_y       = goal_y
        req.spawn_only   = spawn_only
        req.goal_radius  = goal_radius
        return self.call(self._reset_client, req)

    def wait_new_goal(self, prev_seq: int = -1,
                      max_wait_sec: float = 2.0,
                      poll_period_sec: float = 0.05):
        """Block until env_bridge confirms a NEW /goal_pose has arrived
        AFTER `prev_seq`. Uses the monotonic goal_seq counter (instead of
        the sticky new_goal bool which has a race-condition bug).

        Caller pattern:
            seq_before = self.get_observation().goal_seq
            self.reset_episode(...)         # publishes new /goal_pose
            res = self.wait_new_goal(prev_seq=seq_before)  # polls until seq > prev
        """
        deadline = time.time() + max_wait_sec
        while time.time() < deadline:
            res = self.get_observation()
            if res.goal_seq > prev_seq:
                return res
            time.sleep(poll_period_sec)
        return self.get_observation()


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
        self._clearance_threshold = float(cfg_env.get("clearance_threshold", 0.5))

        min_lv = float(cfg_env.get("min_linear_vel", 0.0))  # negative allows reverse
        max_lv = float(cfg_env["max_linear_vel"])
        max_av = float(cfg_env["max_angular_vel"])

        # ── observation layout (42 dims):
        #   [0..35]  min_lidar bins, normalised to [0, 1]
        #   [36]     dist_norm to goal               [0, 1]
        #   [37]     cos(goal_angle in body frame)   [-1, 1]
        #   [38]     sin(goal_angle in body frame)   [-1, 1]
        #   [39]     goal_path_min_lidar (±20° cone) [0, 1]
        #   [40]     prev_lv                          [min_lv, max_lv]
        #   [41]     prev_av                          [-max_av, max_av]
        obs_low  = np.concatenate([
            np.zeros(self._lidar_bins),
            np.array([0.0, -1.0, -1.0, 0.0, min_lv, -max_av]),
        ]).astype(np.float32)
        obs_high = np.concatenate([
            np.ones(self._lidar_bins),
            np.array([1.0,  1.0,  1.0, 1.0, max_lv,  max_av]),
        ]).astype(np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low  = np.array([min_lv, -max_av], dtype=np.float32),
            high = np.array([max_lv,  max_av], dtype=np.float32),
            dtype = np.float32,
        )

        # ── drlnav-style successive goals + adaptive difficulty radius ───────
        self._successive_goals = bool(cfg_env.get("successive_goals", False))
        self._diff_radius      = float(cfg_env.get("difficulty_radius_init", 1.0))
        self._diff_min         = float(cfg_env.get("difficulty_radius_min", 0.5))
        self._diff_max         = float(cfg_env.get("difficulty_radius_max", 4.0))
        self._diff_grow        = float(cfg_env.get("difficulty_grow",   1.01))
        self._diff_shrink      = float(cfg_env.get("difficulty_shrink", 0.99))
        # drlnav task_succeed vs task_fail dispatch: tracks whether the
        # previous episode ended in a goal-reach. If so, reset() spawns a
        # new goal around the robot's current pose (no teleport). Otherwise
        # reset() does a full sim reset (teleport robot to spawn).
        self._was_success      = False

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

        obs = None

        # Snapshot goal_seq BEFORE calling reset_episode — wait_new_goal()
        # then polls until env_bridge's seq advances, guaranteeing the new
        # /goal_pose has propagated AND step_count_ has been reset to 0.
        try:
            seq_before = self._client.get_observation().goal_seq
        except RuntimeError:
            seq_before = 0

        # drlnav task_succeed path: previous episode ended in goal-reach,
        # so the robot stays put — just spawn a new goal at difficulty
        # radius around its current pose. No teleport.
        if self._successive_goals and self._was_success:
            try:
                self._client.reset_episode(
                    spawn_only  = True,
                    goal_radius = self._diff_radius,
                )
                res = self._client.wait_new_goal(prev_seq=seq_before)
                obs = self._parse(res)
            except RuntimeError as e:
                print(f"[env] spawn_only failed: {e}; falling back to full reset.")
                obs = None

        # drlnav task_fail path: previous episode ended in collision/timeout,
        # OR this is the first episode, OR successive goals are disabled.
        # Teleport robot back to spawn and sample a random arena goal.
        # NOTE: removed the old "retry up to 10× if lidar too close" loop.
        # The lidar sits 6 cm behind robot center, so at fixed_spawn (-0.7, 0)
        # the reading to wall 7 (x=-1.125) was 0.36 m, just below the
        # safe_norm threshold of 0.4 m. That triggered 10 cascading resets per
        # episode-end, creating the rapid-goal-change pattern. drlnav does
        # NOT have this check — they trust the policy to navigate even if
        # spawn happens near an obstacle.
        if obs is None:
            try:
                reset_res = self._client.reset_episode(random_pose=True)
                if reset_res.success:
                    res = self._client.wait_new_goal(prev_seq=seq_before)
                    obs = self._parse(res)
                else:
                    time.sleep(0.5)
                    obs = self._parse(self._client.get_observation())
            except RuntimeError as e:
                print(f"[env] reset failed: {e}")
                obs = self._parse(self._client.get_observation())

        # Cleared for the next episode — re-set in step() if this one reaches goal.
        self._was_success = False
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
        reward     = float(res.reward)
        terminated = bool(res.done)
        truncated  = self._step_count >= self._max_steps
        is_success = (res.info == "goal_reached")

        # drlnav adaptive difficulty radius + task_succeed/task_fail dispatch
        # for the NEXT reset(): set _was_success so reset() picks the right
        # branch (spawn_only vs full teleport).
        if is_success:
            self._diff_radius = min(self._diff_max,
                                    self._diff_radius * self._diff_grow)
            self._was_success = True
        elif terminated or truncated:
            self._diff_radius = max(self._diff_min,
                                    self._diff_radius * self._diff_shrink)
            # _was_success stays False → next reset() does full teleport.

        info: dict[str, Any] = {
            "is_success":       is_success,
            "info":             res.info,
            "progress_reward":  float(res.progress_reward),
            "front_min":        float(res.front_min),
            "progress_gated":   bool(res.progress_gated),
            "diff_radius":      self._diff_radius,
        }
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
            # Diagnostic so we can tell timeouts apart from C++ terminations.
            print(f"[env] truncation: step_count={self._step_count} "
                  f"max={self._max_steps} info='{res.info}' "
                  f"reward={reward:.1f}")

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._client.destroy_node()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _parse(res) -> np.ndarray:
        return np.array(res.observation, dtype=np.float32)
