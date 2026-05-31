#!/usr/bin/env python3
"""Diagnostic for build_dataset.py — prints per-episode counts.

Walks the bag the same way build_dataset does but logs:
  - how many /cmd_vel messages each episode received
  - how many were dropped because /scan / /robot_world_pose / /goal_pose
    weren't ready yet
  - which episodes were marked successful
"""

from __future__ import annotations

import argparse
import math
import sys

sys.path.insert(0, "/bc")
from build_dataset import read_bag, quat_yaw

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--goal-tolerance", type=float, default=0.20)
    args = p.parse_args()

    latest_scan  = None
    latest_world = None
    latest_goal  = None
    current_ep   = -1
    ep_cmd_added = 0
    ep_cmd_dropped = 0
    ep_min_dist  = float("inf")

    per_ep = []  # list of (ep_id, n_added, n_dropped, min_dist)

    def finalize():
        nonlocal ep_cmd_added, ep_cmd_dropped, ep_min_dist
        if current_ep >= 0:
            per_ep.append((current_ep, ep_cmd_added, ep_cmd_dropped, ep_min_dist))
        ep_cmd_added = 0
        ep_cmd_dropped = 0
        ep_min_dist = float("inf")

    for topic, msg, _t in read_bag(args.bag):
        if topic == "/scan":
            latest_scan = True
        elif topic == "/robot_world_pose":
            p_ = msg.pose
            latest_world = (float(p_.position.x), float(p_.position.y),
                            quat_yaw(p_.orientation.z, p_.orientation.w))
            if latest_goal is not None:
                gx, gy = latest_goal
                d = math.hypot(latest_world[0] - gx, latest_world[1] - gy)
                if d < ep_min_dist:
                    ep_min_dist = d
        elif topic == "/goal_pose":
            finalize()
            current_ep += 1
            latest_goal = (float(msg.pose.position.x),
                           float(msg.pose.position.y))
        elif topic == "/cmd_vel":
            if latest_scan is None or latest_world is None or latest_goal is None:
                ep_cmd_dropped += 1
            else:
                ep_cmd_added += 1
    finalize()

    arr = np.array([(e, a, d, m) for (e, a, d, m) in per_ep],
                   dtype=[("ep", int), ("added", int), ("dropped", int), ("min_dist", float)])

    successful = arr["min_dist"] <= args.goal_tolerance
    print(f"Total episodes:       {len(arr)}")
    print(f"Successful (min_dist≤{args.goal_tolerance}): {successful.sum()}")
    print()
    print(f"cmd_vel ADDED   — sum {arr['added'].sum():>7d}   "
          f"mean {arr['added'].mean():.1f}   median {int(np.median(arr['added']))}   "
          f"max {arr['added'].max()}")
    print(f"cmd_vel DROPPED — sum {arr['dropped'].sum():>7d}   "
          f"mean {arr['dropped'].mean():.1f}   median {int(np.median(arr['dropped']))}   "
          f"max {arr['dropped'].max()}")
    print()
    print("First 10 episodes (ep, added, dropped, min_dist):")
    for row in arr[:10]:
        print(f"  ep={row['ep']:3d}  added={row['added']:4d}  dropped={row['dropped']:4d}  "
              f"min_dist={row['min_dist']:.3f}")
    print()
    print("Last 5 episodes:")
    for row in arr[-5:]:
        print(f"  ep={row['ep']:3d}  added={row['added']:4d}  dropped={row['dropped']:4d}  "
              f"min_dist={row['min_dist']:.3f}")


if __name__ == "__main__":
    main()
