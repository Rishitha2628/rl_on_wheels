#!/usr/bin/env python3
"""Convert a BC-format NPZ into imitation.data.types.Trajectory pickle.

The BC dataset stores flat (obs, action, ep_id) per timestep. AIRL needs
trajectories — variable-length lists of (T+1 obs, T acts, terminal) — so
each Nav2 episode becomes one Trajectory.

Frame stacking: with --frame-stack k, each per-step observation becomes
the concatenation of the last k single-frame obs within the episode
(padded by repeating the first obs for the first k-1 steps). This gives
the policy temporal context — same trick as bc/train.py used to lift BC
from 52 % to 62 % on stage 5.

Caveat: the BC NPZ has N obs and N actions per episode (the obs at each
/cmd_vel time), but no "next state" for the final action. We drop the
last action so an N-pair episode produces T = N-1 transitions with T+1
= N obs (which is what Trajectory expects).

Usage (inside container):
    python3 /airl/convert_demos.py \\
        --npz /demos/stage5_bc_6m.npz \\
        --out /demos/stage5_trajectories_fs4.pkl \\
        --frame-stack 4
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from imitation.data.types import Trajectory


def stack_episode(ep_obs: np.ndarray, k: int) -> np.ndarray:
    """Stack last k obs at each step within a single episode.

    Output[t] = concat(obs[t-(k-1)], ..., obs[t-1], obs[t]) — oldest left,
    newest right. The first k-1 steps repeat the first obs to keep shape
    fixed.
    """
    N, D = ep_obs.shape
    out = np.empty((N, k * D), dtype=np.float32)
    for i in range(N):
        frames = []
        for j in range(k):
            src = max(0, i - (k - 1 - j))
            frames.append(ep_obs[src])
        out[i] = np.concatenate(frames)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="BC-format NPZ from bc/build_dataset.py")
    p.add_argument("--out", required=True, help="output pickle path")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="number of consecutive obs to stack per step (1 = off)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.npz)
    obs   = d["obs"].astype(np.float32)
    acts  = d["action"].astype(np.float32)
    ep_id = d["ep_id"].astype(np.int32)
    k = int(args.frame_stack)

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
        if k > 1:
            ep_obs = stack_episode(ep_obs, k)
        traj = Trajectory(
            obs=ep_obs,                # T+1 = N obs (each k*D dim if stacked)
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
    obs_dim = trajectories[0].obs.shape[1] if trajectories else 0
    print(f"saved {len(trajectories)} trajectories ({total_steps} transitions, "
          f"mean episode length {mean_len:.1f}, obs_dim={obs_dim}) → {out_path}")
    if n_skipped:
        print(f"skipped {n_skipped} episodes with <2 transitions")


if __name__ == "__main__":
    main()
