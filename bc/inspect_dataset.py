#!/usr/bin/env python3
"""Compare action / obs distributions between two BC NPZs.

Use to diagnose DAgger quality regressions: if the new DAgger pairs have
wildly different action stats than the original demos, the relabeling
went badly. Particularly check what Nav2 said to do when obs[39]
(goal_path_min, ie 'is the path to goal blocked?') is small.

Usage:
    python3 /bc/inspect_dataset.py /demos/stage4_bc.npz /demos/dagger_iter1.npz
"""

import sys

import numpy as np


def stats(name: str, path: str) -> None:
    d = np.load(path)
    obs, act = d["obs"], d["action"]
    print(f"\n── {name}: {path} ──")
    print(f"  n_pairs = {len(obs)}")
    print(f"  lin_vel  mean={act[:,0].mean():+.3f}  std={act[:,0].std():.3f}  "
          f"min={act[:,0].min():+.3f}  max={act[:,0].max():+.3f}")
    print(f"  ang_vel  mean={act[:,1].mean():+.3f}  std={act[:,1].std():.3f}  "
          f"min={act[:,1].min():+.3f}  max={act[:,1].max():+.3f}")

    # Subset: states where the path to goal is "blocked" (obs[39] < 0.15
    # means closest lidar reading in goal direction is < 0.15 * 3.5 m = 0.52 m).
    blocked = obs[:, 39] < 0.15
    n_blocked = int(blocked.sum())
    if n_blocked > 0:
        print(f"  ── 'path-to-goal blocked' subset (obs[39] < 0.15): "
              f"{n_blocked} pairs ({100*n_blocked/len(obs):.1f}%) ──")
        print(f"      lin_vel mean={act[blocked,0].mean():+.3f}  "
              f"frac(lin>0.10)={float((act[blocked,0]>0.10).mean()):.2f}")
        print(f"      ang_vel mean={act[blocked,1].mean():+.3f}  "
              f"mean(|ang|)={float(np.abs(act[blocked,1]).mean()):.3f}")

    # Subset: states with ANY lidar bin very close (collision-imminent).
    closest = obs[:, :36].min(axis=1)
    very_close = closest < 0.10   # 0.10 * 3.5 = 0.35 m
    n_close = int(very_close.sum())
    if n_close > 0:
        print(f"  ── 'imminent collision' subset (min lidar < 0.10): "
              f"{n_close} pairs ({100*n_close/len(obs):.1f}%) ──")
        print(f"      lin_vel mean={act[very_close,0].mean():+.3f}  "
              f"frac(lin>0.10)={float((act[very_close,0]>0.10).mean()):.2f}")
        print(f"      ang_vel mean={act[very_close,1].mean():+.3f}  "
              f"mean(|ang|)={float(np.abs(act[very_close,1]).mean()):.3f}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: inspect_dataset.py <npz1> [npz2 ...]")
        sys.exit(1)
    for i, path in enumerate(sys.argv[1:]):
        stats(f"set{i+1}", path)


if __name__ == "__main__":
    main()
