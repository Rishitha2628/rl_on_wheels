#!/usr/bin/env python3
"""Combine demonstrations recorded on two arenas with different LiDAR scales.

    python3 bc/merge_arenas.py --a demos/stage4_dagger_agg2.npz --a-range 3.5 \
        --b demos/stage11_dagger_agg2_r35.npz --out demos/both_arenas_r35.npz

The 5x5 arena recorded with max_lidar_range 3.5 and the 7x7 with 6.0, and that
constant normalises the goal distance as well as the LiDAR
(env_bridge_node.cpp). Concatenating the raw arrays would therefore train a
policy on observations where the same number means different distances.

We rebase the 3.5 m set onto the 6.0 m convention. A bin already at 1.0 was a
no-return at 3.5 m and stays a no-return (1.0); anything shorter is a real
measurement and becomes d/6.0. That is exactly the convention produced by
bc/clip_lidar_range.py, so the merged set represents one 3.5 m sensor
throughout and should be evaluated with --lidar-max-range 3.5.
"""
import argparse
import math

import numpy as np

BINS, RMAX, GOAL_HALF_WINDOW = 36, 6.0, 2


def rebase(obs: np.ndarray, src_range: float) -> np.ndarray:
    o = obs.copy()
    b = o[:, :BINS].astype(np.float64)
    no_return = b >= 0.999                       # saturated at the old max
    d = b * src_range                            # → metres
    o[:, :BINS] = np.where(no_return, 1.0, d / RMAX).astype(np.float32)

    # obs[36] is the goal distance, normalised by the same constant. Values
    # that were saturated carry only "at least src_range"; we map them to
    # src_range, which under-states them. This is the one lossy step.
    o[:, BINS] = np.minimum(o[:, BINS].astype(np.float64) * src_range, RMAX) / RMAX

    ang = np.arctan2(o[:, BINS + 2], o[:, BINS + 1])
    pos = np.mod(ang + 2 * math.pi, 2 * math.pi)
    centre = (pos / (2 * math.pi) * BINS).astype(int) % BINS
    idx = (centre[:, None] + np.arange(-GOAL_HALF_WINDOW, GOAL_HALF_WINDOW + 1)[None, :]) % BINS
    o[:, BINS + 3] = np.take_along_axis(o[:, :BINS], idx, axis=1).min(axis=1)
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="set to rebase")
    p.add_argument("--a-range", type=float, required=True)
    p.add_argument("--b", required=True, help="set already on the 6.0 convention")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    oa = rebase(A["obs"].astype(np.float32), args.a_range)
    ob = B["obs"].astype(np.float32)
    print(f"  A {args.a}: {len(oa)} transitions, rebased {args.a_range} m -> {RMAX} m")
    print(f"    saturated bins kept as no-return: {(A['obs'][:, :BINS] >= 0.999).mean():.3f}")
    print(f"  B {args.b}: {len(ob)} transitions")

    # Keep episode ids disjoint so downstream grouping stays correct.
    ea = A["ep_id"].astype(np.int64)
    eb = B["ep_id"].astype(np.int64) + int(ea.max()) + 1
    np.savez_compressed(
        args.out,
        obs=np.concatenate([oa, ob]),
        action=np.concatenate([A["action"], B["action"]]).astype(np.float32),
        ep_id=np.concatenate([ea, eb]).astype(np.int32),
    )
    print(f"  wrote {args.out}: {len(oa) + len(ob)} transitions, "
          f"{len(set(ea.tolist())) + len(set(eb.tolist()))} episodes")


if __name__ == "__main__":
    main()
