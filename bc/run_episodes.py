#!/usr/bin/env python3
"""Drive autonomous episodes while Nav2 navigates.

Each iteration:
  1. Calls /reset_episode → reset_node teleports the robot to a fresh spawn
     and publishes a new /goal_pose
  2. goal_forwarder_node hands that goal to Nav2 (already running)
  3. Waits until EITHER:
       - the robot reaches the goal (distance < goal_tolerance), OR
       - the episode times out (--episode-seconds)
  4. Triggers the next reset.

Usage (inside container, after sim + Nav2 are running):
    python3 /bc/run_episodes.py --episodes 200 --episode-seconds 30
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from tb3_rl_bridge.srv import ResetEpisode


class EpisodeRunner(Node):
    def __init__(self, _unused_goal_tolerance: float = 0.0) -> None:
        super().__init__("episode_runner")

        self.reset_client = self.create_client(ResetEpisode, "/reset_episode")
        self.get_logger().info("Waiting for /reset_episode service...")
        self.reset_client.wait_for_service()
        self.get_logger().info("/reset_episode service ready.")

        # /nav_episode_result is published by goal_forwarder_node when Nav2's
        # action completes. True = SUCCEEDED, False = ABORTED/CANCELED.
        self._nav_result: bool | None = None
        self.create_subscription(Bool, "/nav_episode_result",
                                 self._on_nav_result, 10)

    def _on_nav_result(self, msg: Bool) -> None:
        self._nav_result = msg.data

    def trigger_reset(self) -> bool:
        # Clear pending result so we don't catch the cancellation noise from
        # preempting the previous goal.
        self._nav_result = None

        req = ResetEpisode.Request()
        req.random_pose = True
        req.spawn_only  = False
        fut = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if not fut.done():
            self.get_logger().warn("reset_episode timed out")
            return False
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().warn(f"reset_episode failed: {res}")
            return False

        # The reset itself may have triggered a cancellation of the previous
        # Nav2 goal — discard that result so we only see the NEW episode's.
        self._nav_result = None
        return True

    def wait_for_episode_end(self, timeout: float) -> str:
        """Wait for Nav2 SUCCEEDED, ABORTED/CANCELED, or timeout."""
        t_start = time.time()
        poll = 0.1
        while time.time() - t_start < timeout:
            rclpy.spin_once(self, timeout_sec=poll)
            if self._nav_result is True:
                return "success"
            if self._nav_result is False:
                return "abort"
        return "timeout"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",        type=int,   default=200)
    p.add_argument("--episode-seconds", type=float, default=30.0,
                   help="max time per episode before next reset")
    args = p.parse_args()

    rclpy.init()
    runner = EpisodeRunner()

    n_success = 0
    n_abort   = 0
    n_timeout = 0
    for ep in range(1, args.episodes + 1):
        ok = runner.trigger_reset()
        if not ok:
            runner.get_logger().warn(f"[ep {ep}] reset failed — skipping")
            time.sleep(1.0)
            continue
        result = runner.wait_for_episode_end(args.episode_seconds)
        if result == "success":
            n_success += 1
        elif result == "abort":
            n_abort += 1
        else:
            n_timeout += 1
        runner.get_logger().info(
            f"[ep {ep:4d}/{args.episodes}] {result}  "
            f"({n_success}✓ {n_abort}✗ {n_timeout}⌛)")

    runner.get_logger().info(
        f"All episodes complete. Success: {n_success}/{args.episodes} "
        f"({100*n_success/max(args.episodes,1):.1f}%)")
    runner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
