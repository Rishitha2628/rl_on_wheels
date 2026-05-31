#!/usr/bin/env python3
"""Convert a ros2 bag of Nav2 demonstrations into a BC training NPZ.

Usage (inside container):
    python3 /bc/build_dataset.py \
        --bag /demos/stage4_nav2 \
        --out /demos/stage4_bc.npz \
        --lidar-bins 36 --max-range 3.5 --goal-tolerance 0.20

The bag is expected to contain at least:
    /scan        sensor_msgs/msg/LaserScan
    /odom        nav_msgs/msg/Odometry
    /goal_pose   geometry_msgs/msg/PoseStamped
    /cmd_vel     geometry_msgs/msg/Twist

For each /cmd_vel message at time t we pair it with the most-recent
/scan, /odom and /goal_pose seen up to t and compute the same 40-dim
observation that env_bridge_node produces. Transitions are grouped into
episodes by /goal_pose changes; only successful episodes (robot reached
within goal_tolerance) are kept.

Output NPZ keys:
    obs        (N, obs_dim)  float32
    action     (N, 2)        float32  — [linear_vel, angular_vel]
    ep_id      (N,)          int32    — episode index per transition
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


# Lazy imports of ROS-specific modules so the file can at least be imported
# outside the container (e.g. for the test in build_dataset_test.py).
def _import_rosbag():
    from rclpy.serialization import deserialize_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message
    return deserialize_message, SequentialReader, StorageOptions, \
        ConverterOptions, get_message


# ── observation computation (mirrors env_bridge_node) ─────────────────────────
def downsample_lidar(ranges: np.ndarray, n_bins: int, max_range: float) -> np.ndarray:
    """min-pool the raw 360-sample scan into n_bins, normalize by max_range."""
    n_src = ranges.shape[0]
    per_bin = n_src // n_bins
    out = np.empty(n_bins, dtype=np.float32)
    for b in range(n_bins):
        chunk = ranges[b * per_bin:(b + 1) * per_bin]
        chunk = np.where(np.isfinite(chunk), chunk, max_range)
        chunk = np.clip(chunk, 0.0, max_range)
        out[b] = chunk.min()
    return out / max_range


def quat_yaw(qz: float, qw: float) -> float:
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def compute_obs(scan_ranges: np.ndarray, robot_x: float, robot_y: float,
                robot_yaw: float, goal_x: float, goal_y: float,
                prev_lin_vel: float, prev_ang_vel: float,
                lidar_bins: int, max_range: float
                ) -> np.ndarray:
    """Compute the 42-dim observation that matches env_bridge_node exactly.

    Layout (must match env_bridge_node.cpp / build_observation()):
        [0..35]  normalised lidar bins (already normalised by downsample_lidar)
        [36]     dist_norm                       (∈ [0, 1])
        [37]     cos(goal_angle_body)
        [38]     sin(goal_angle_body)
        [39]     goal_path_min — min lidar reading in a ±20° cone around the
                 GOAL direction (NOT around 'forward'). Asks "is the path TO
                 the goal blocked?"
        [40]     prev linear  velocity (m/s, unnormalised)
        [41]     prev angular velocity (rad/s, unnormalised)
    """
    lidar = downsample_lidar(scan_ranges, lidar_bins, max_range)

    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.hypot(dx, dy)
    dist_norm = min(dist / max_range, 1.0)

    goal_body = math.atan2(dy, dx) - robot_yaw
    while goal_body >  math.pi: goal_body -= 2 * math.pi
    while goal_body < -math.pi: goal_body += 2 * math.pi

    # ── goal_path_min — match env_bridge_node:build_observation() exactly ────
    # Lidar bin layout: bin 0 covers 0°-(360°/lidar_bins), bin index increases
    # counter-clockwise. So a goal body-angle θ ∈ [-π, π] maps to bin index
    # round(θ_pos / 2π * n_bins) where θ_pos = θ mod 2π.
    goal_body_pos = (goal_body + 2 * math.pi) % (2 * math.pi)
    center_bin = int(goal_body_pos / (2 * math.pi) * lidar_bins) % lidar_bins
    goal_half_window = 2   # ±2 bins (5 bins total) — match env_bridge
    goal_path_min = 1.0
    for b in range(-goal_half_window, goal_half_window + 1):
        bin_idx = (center_bin + b + lidar_bins) % lidar_bins
        goal_path_min = min(goal_path_min, float(lidar[bin_idx]))

    # Note: prev_lin_vel / prev_ang_vel are intentionally OMITTED from the
    # BC observation. Including them creates a feedback loop where the policy
    # just echoes back the previous action (BC learns the cheap shortcut
    # output ≈ prev_action). Without them, BC must use lidar + goal_path_min
    # to choose actions, which generalises better.
    _ = prev_lin_vel; _ = prev_ang_vel   # accepted but unused
    return np.concatenate([
        lidar.astype(np.float32),
        np.array([dist_norm,
                  math.cos(goal_body), math.sin(goal_body),
                  goal_path_min], dtype=np.float32),
    ])


# ── bag reading ───────────────────────────────────────────────────────────────
TOPIC_TYPES = {
    "/scan":             "sensor_msgs/msg/LaserScan",
    "/odom":             "nav_msgs/msg/Odometry",       # used only for prev_lin_vel
    "/robot_world_pose": "geometry_msgs/msg/PoseStamped",  # robot pose in map frame
    "/goal_pose":        "geometry_msgs/msg/PoseStamped",
    "/cmd_vel":          "geometry_msgs/msg/Twist",
}


def read_bag(bag_path: str):
    """Yield (topic, msg, t_ns) in time order from a sqlite3 ros2 bag."""
    deserialize, SeqReader, StorageOpts, ConvOpts, get_msg = _import_rosbag()
    reader = SeqReader()
    reader.open(
        StorageOpts(uri=bag_path, storage_id="sqlite3"),
        ConvOpts(input_serialization_format="cdr",
                 output_serialization_format="cdr"),
    )
    # Pre-resolve message classes for the topics we care about.
    msg_cls = {t: get_msg(typ) for t, typ in TOPIC_TYPES.items()}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in msg_cls:
            continue
        yield topic, deserialize(data, msg_cls[topic]), t_ns


# ── main pipeline ─────────────────────────────────────────────────────────────
def build_dataset(bag_path: str, lidar_bins: int, max_range: float,
                  goal_tolerance: float) -> dict[str, np.ndarray]:
    # State trackers — populated as we walk the bag in time order.
    latest_scan = None
    latest_world = None    # (x, y, yaw) from /robot_world_pose
    # prev_lin_vel / prev_ang_vel are the PREVIOUS /cmd_vel (matches what
    # env_bridge_node stores as last_lv_/last_av_). We update them AFTER each
    # /cmd_vel is paired into a transition, so the NEXT /cmd_vel's obs uses
    # this one as its 'prev'.
    prev_lin_vel = 0.0
    prev_ang_vel = 0.0
    latest_goal = None     # (gx, gy)
    current_ep_id = -1

    all_obs:     list[np.ndarray] = []
    all_act:     list[np.ndarray] = []
    all_ep:      list[int]         = []
    ep_buf_obs:  list[np.ndarray] = []
    ep_buf_act:  list[np.ndarray] = []
    ep_min_dist  = float("inf")  # closest the robot came to its goal in this ep

    def finalize_episode():
        nonlocal ep_buf_obs, ep_buf_act, ep_min_dist, current_ep_id
        if not ep_buf_obs:
            return
        # Success = robot got within goal_tolerance at any point during the ep
        if ep_min_dist <= goal_tolerance:
            all_obs.extend(ep_buf_obs)
            all_act.extend(ep_buf_act)
            all_ep.extend([current_ep_id] * len(ep_buf_obs))
        ep_buf_obs = []
        ep_buf_act = []
        ep_min_dist = float("inf")

    for topic, msg, _t in read_bag(bag_path):
        if topic == "/scan":
            latest_scan = np.asarray(msg.ranges, dtype=np.float32)

        elif topic == "/odom":
            # No longer used — kept here to avoid logging spam if it appears.
            pass

        elif topic == "/robot_world_pose":
            p = msg.pose
            yaw = quat_yaw(p.orientation.z, p.orientation.w)
            latest_world = (float(p.position.x), float(p.position.y), yaw)

            # Track closest approach to goal for success detection
            if latest_goal is not None and latest_world is not None:
                gx, gy = latest_goal
                d = math.hypot(latest_world[0] - gx, latest_world[1] - gy)
                if d < ep_min_dist:
                    ep_min_dist = d

        elif topic == "/goal_pose":
            # New goal → previous episode ends.
            finalize_episode()
            current_ep_id += 1
            latest_goal = (float(msg.pose.position.x),
                           float(msg.pose.position.y))
            # Reset prev-action trackers — first transition of new episode
            # should see prev_lv/prev_av = 0, matching env_bridge's behaviour
            # at episode start.
            prev_lin_vel = 0.0
            prev_ang_vel = 0.0

        elif topic == "/cmd_vel":
            if (latest_scan is None or latest_world is None
                    or latest_goal is None):
                continue
            rx, ry, ryaw = latest_world
            gx, gy = latest_goal
            # Build obs using the PREVIOUS cmd_vel (matches env_bridge's last_lv_/last_av_).
            obs = compute_obs(latest_scan, rx, ry, ryaw, gx, gy,
                              prev_lin_vel, prev_ang_vel,
                              lidar_bins, max_range)
            act = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)
            ep_buf_obs.append(obs)
            ep_buf_act.append(act)
            # Now THIS cmd_vel becomes the "prev" for the next transition.
            prev_lin_vel = float(msg.linear.x)
            prev_ang_vel = float(msg.angular.z)

    finalize_episode()

    if not all_obs:
        raise RuntimeError(
            "No successful episodes found. Check that the bag contains the "
            "expected topics and that the robot actually reached its goals.")

    return {
        "obs":    np.stack(all_obs).astype(np.float32),
        "action": np.stack(all_act).astype(np.float32),
        "ep_id":  np.array(all_ep, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True, help="path to ros2 bag directory")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--lidar-bins",     type=int,   default=36)
    p.add_argument("--max-range",      type=float, default=3.5)
    p.add_argument("--goal-tolerance", type=float, default=0.20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"reading bag: {args.bag}")
    data = build_dataset(args.bag, args.lidar_bins, args.max_range,
                         args.goal_tolerance)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    n_ep = int(data["ep_id"].max()) + 1 if len(data["ep_id"]) else 0
    print(f"saved {len(data['obs'])} transitions from {n_ep} successful "
          f"episodes → {out_path}")


if __name__ == "__main__":
    main()
