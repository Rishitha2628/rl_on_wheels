#!/usr/bin/env python3
"""Convert a BC-format NPZ into imitation.data.types.Trajectory pickle.

The BC dataset stores flat (obs, action, ep_id) per timestep. AIRL needs
trajectories — variable-length lists of (T+1 obs, T acts, terminal) — so
each Nav2 episode becomes one Trajectory.

Caveat: the BC NPZ has N obs and N actions per episode (the obs at each
/cmd_vel time), but no "next state" for the final action. We drop the
last action so an N-pair episode produces T = N-1 transitions with T+1
= N obs (which is what Trajectory expects).

Usage (inside container):
    python3 /airl/convert_demos.py \\
        --npz /demos/stage11_bc_6m.npz \\
        --out /demos/stage11_trajectories.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from imitation.data.types import Trajectory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="BC-format NPZ from bc/build_dataset.py")
    p.add_argument("--out", required=True, help="output pickle path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.npz)
    obs   = d["obs"].astype(np.float32)
    acts  = d["action"].astype(np.float32)
    ep_id = d["ep_id"].astype(np.int32)

    trajectories: list[Trajectory] = []
    n_skipped = 0
    for ep in np.unique(ep_id):
        idx = np.where(ep_id == ep)[0]
        # Need at least 2 obs to form a transition.
        if len(idx) < 2:
            n_skipped += 1
            continue
        ep_obs  = obs[idx]            # (N, D)
        ep_acts = acts[idx]           # (N, A)
        traj = Trajectory(
            obs=ep_obs,                # T+1 = N obs
            acts=ep_acts[:-1],         # T = N-1 actions
            infos=None,
            terminal=True,             # all Nav2 demos are successful episodes
        )
        trajectories.append(traj)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(trajectories, f)

    total_steps = sum(len(t.acts) for t in trajectories)
    mean_len = total_steps / max(1, len(trajectories))
    print(f"saved {len(trajectories)} trajectories ({total_steps} transitions, "
          f"mean episode length {mean_len:.1f}) → {out_path}")
    if n_skipped:
        print(f"skipped {n_skipped} episodes with <2 transitions")


if __name__ == "__main__":
    main()
