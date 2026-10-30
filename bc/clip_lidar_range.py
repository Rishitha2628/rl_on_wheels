#!/usr/bin/env python3
"""Clip recorded demonstrations to a shorter LiDAR range.

    python3 bc/clip_lidar_range.py --in demos/x.npz --out demos/x_r35.npz --range 3.5

Why not just set max_lidar_range in the config: that constant normalises the
goal distance as well as the LiDAR (env_bridge_node.cpp), so lowering it would
saturate every goal beyond the new range and confound a range experiment with
a goal-encoding change. Clipping the data leaves the /6.0 normalisation and
obs[36] untouched.

The transform matches rl/envs/substrate_wrappers.LidarDegradation exactly, so
a policy trained on the output can be evaluated with --lidar-max-range and see
the same distribution: a reading beyond the new range is a no-return, which
the bridge represents as max range (1.0 normalised).
"""
import argparse
import math

import numpy as np

BINS, RMAX, GOAL_HALF_WINDOW = 36, 6.0, 2


def clip(obs: np.ndarray, max_range_m: float) -> np.ndarray:
    o = obs.copy()
    d = o[:, :BINS].astype(np.float64) * RMAX              # → metres
    d = np.where(d > max_range_m, RMAX, d)                 # no return → max
    o[:, :BINS] = (d / RMAX).astype(np.float32)

    # obs[39] is the min over a +/-2-bin cone about the goal bearing and must be
    # recomputed from the clipped bins, or the policy keeps an undegraded view.
    ang = np.arctan2(o[:, BINS + 2], o[:, BINS + 1])
    pos = np.mod(ang + 2 * math.pi, 2 * math.pi)
    centre = (pos / (2 * math.pi) * BINS).astype(int) % BINS
    idx = (centre[:, None] + np.arange(-GOAL_HALF_WINDOW, GOAL_HALF_WINDOW + 1)[None, :]) % BINS
    o[:, BINS + 3] = np.take_along_axis(o[:, :BINS], idx, axis=1).min(axis=1)
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--range", type=float, default=3.5)
    a = p.parse_args()

    d = np.load(a.src)
    obs = d["obs"].astype(np.float32)
    clipped = clip(obs, a.range)

    frac = float((obs[:, :BINS] * RMAX > a.range).mean())
    print(f"  {len(obs)} transitions, {100 * frac:.1f}% of bins beyond {a.range} m")
    print(f"  bins  before: [{obs[:, :BINS].min():.3f}, {obs[:, :BINS].max():.3f}]"
          f"   after: [{clipped[:, :BINS].min():.3f}, {clipped[:, :BINS].max():.3f}]")
    print(f"  obs[36] unchanged: {np.array_equal(obs[:, 36], clipped[:, 36])}")

    out = {k: d[k] for k in d.keys()}
    out["obs"] = clipped
    np.savez_compressed(a.dst, **out)
    print(f"  wrote {a.dst}")


if __name__ == "__main__":
    main()
