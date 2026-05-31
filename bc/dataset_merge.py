#!/usr/bin/env python3
"""Concatenate multiple BC/DAgger NPZ files into a single training set.

Each input NPZ has keys: obs (N, D), action (N, 2), ep_id (N,) int32.
Episode ids are remapped to a globally unique range so they don't collide.

Usage:
    python3 /bc/dataset_merge.py \
        --inputs /demos/stage4_bc.npz /demos/dagger_iter1.npz \
        --out    /demos/stage4_dagger_iter1.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="paths to input NPZs (order matters: later files "
                        "are typically newer DAgger iters)")
    p.add_argument("--out", required=True, help="output NPZ path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    obs_chunks:    list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    ep_chunks:     list[np.ndarray] = []
    ep_offset = 0
    for path in args.inputs:
        d = np.load(path)
        o, a, e = d["obs"], d["action"], d["ep_id"]
        e = e.astype(np.int32) + ep_offset
        obs_chunks.append(o.astype(np.float32))
        action_chunks.append(a.astype(np.float32))
        ep_chunks.append(e)
        n_eps_here = int(d["ep_id"].max()) + 1 if len(d["ep_id"]) else 0
        ep_offset += n_eps_here
        print(f"  {path}: {len(o):6d} pairs, {n_eps_here:4d} episodes")

    obs    = np.concatenate(obs_chunks,    axis=0)
    action = np.concatenate(action_chunks, axis=0)
    ep_id  = np.concatenate(ep_chunks,     axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        obs=obs, action=action, ep_id=ep_id)
    print(f"\nmerged {len(obs)} pairs across {ep_offset} episodes → {out_path}")


if __name__ == "__main__":
    main()
