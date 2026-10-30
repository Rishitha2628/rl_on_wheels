#!/usr/bin/env python3
"""Resample a demonstration set from 10 Hz to a lower decision rate.

    python3 bc/resample_rate.py --in demos/stage11_dagger_agg2.npz \
        --out demos/stage11_dagger_agg2_2p5hz.npz --every 4

Why this exists
---------------
The factorial varies frame stacking AND control rate, but every policy was
trained on 10 Hz demonstrations and only *evaluated* at 2.5 Hz. With four
stacked frames that means the network saw frames 0.1 s apart in training and
0.4 s apart at evaluation -- an input distribution it was never trained on.

So the measured "4 frames + slow control is terrible" may be nothing more than
distance from the training conditions, rather than anything about what the
policy learned. Keeping every Nth transition makes consecutive rows N*0.1 s
apart, so a policy trained on this data and evaluated with action_repeat=N sees
the same frame spacing in both. bc/train.py's stack_frames() builds stacks from
consecutive rows within an episode, which is what makes this work.

The action kept at each retained step is the one the expert took there, which
is also the action that gets held for N bridge steps at evaluation time.
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=4,
                   help="keep every Nth transition; 4 turns 10 Hz into 2.5 Hz")
    args = p.parse_args()

    d = np.load(args.inp)
    obs, act, ep = d["obs"], d["action"], d["ep_id"]

    keep = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]          # already in step order
        keep.append(idx[::args.every])
    keep = np.concatenate(keep)
    keep.sort()

    np.savez_compressed(args.out, obs=obs[keep], action=act[keep], ep_id=ep[keep])
    print(f"  {args.inp}")
    print(f"    {obs.shape[0]:,} transitions, {len(np.unique(ep))} episodes")
    print(f"  -> {args.out}  (every {args.every})")
    print(f"    {len(keep):,} transitions, {len(np.unique(ep[keep]))} episodes"
          f"  ({100*len(keep)/obs.shape[0]:.1f}% kept)")


if __name__ == "__main__":
    main()
