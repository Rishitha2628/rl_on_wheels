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
                prev_lin_vel: float, lidar_bins: int, max_range: float
                ) -> np.ndarray:
    lidar = downsample_lidar(scan_ranges, lidar_bins, max_range)

    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.hypot(dx, dy)
    dist_norm = min(dist / max_range, 1.0)

    goal_body = math.atan2(dy, dx) - robot_yaw
    while goal_body >  math.pi: goal_body -= 2 * math.pi
    while goal_body < -math.pi: goal_body += 2 * math.pi

    return np.concatenate([
        lidar.astype(np.float32),
        np.array([dist_norm, math.cos(goal_body), math.sin(goal_body),
                  prev_lin_vel], dtype=np.float32),
    ])


# ── bag reading ───────────────────────────────────────────────────────────────
TOPIC_TYPES = {
    "/scan":      "sensor_msgs/msg/LaserScan",
    "/odom":      "nav_msgs/msg/Odometry",
    "/goal_pose": "geometry_msgs/msg/PoseStamped",
    "/cmd_vel":   "geometry_msgs/msg/Twist",
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
    latest_odom = None     # (x, y, yaw, lin_vel)
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
            p = msg.pose.pose
            yaw = quat_yaw(p.orientation.z, p.orientation.w)
            lin = float(msg.twist.twist.linear.x)
            latest_odom = (float(p.position.x), float(p.position.y), yaw, lin)

            # Track closest approach to goal for success detection
            if latest_goal is not None and latest_odom is not None:
                gx, gy = latest_goal
                d = math.hypot(latest_odom[0] - gx, latest_odom[1] - gy)
                if d < ep_min_dist:
                    ep_min_dist = d

        elif topic == "/goal_pose":
            # New goal → previous episode ends.
            finalize_episode()
            current_ep_id += 1
            latest_goal = (float(msg.pose.position.x),
                           float(msg.pose.position.y))

        elif topic == "/cmd_vel":
            if latest_scan is None or latest_odom is None or latest_goal is None:
                continue
            rx, ry, ryaw, prev_lin = latest_odom
            gx, gy = latest_goal
            obs = compute_obs(latest_scan, rx, ry, ryaw, gx, gy,
                              prev_lin, lidar_bins, max_range)
            act = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)
            ep_buf_obs.append(obs)
            ep_buf_act.append(act)

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
