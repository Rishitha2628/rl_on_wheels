#!/usr/bin/env python3
"""Prove the training and evaluation paths build the same masked input.

    python3 bc/verify_history_mask.py

Training stacks offline with stack_frames() over a whole episode, then masks.
Evaluation fills a deque one step at a time, then masks. If those two disagree
the policy is scored on inputs it never trained on, and the result would look
like an effect rather than a bug -- the exact failure this paper is about. So
the two paths are reconstructed here and compared bit for bit on real demo
episodes, for every history setting.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from bc import history_mask
from bc.train import stack_frames

# Runs both on the host and inside the sim container, where the repo is mounted
# at / and the demos at /demos.
DATASET = next(p for p in ("/demos/stage11_dagger_agg2.npz",
                           os.path.join(_HERE, "demos/stage11_dagger_agg2.npz"))
               if os.path.exists(p))
K = 4


def eval_path(obs_ep: np.ndarray, keep: str) -> np.ndarray:
    """Reproduce bc/eval.py: seed the ring buffer with the first obs, then
    append each step before acting, masking the stack every time."""
    base = obs_ep.shape[1]
    buf: deque = deque(maxlen=K)
    for _ in range(K):
        buf.append(obs_ep[0])
    rows = []
    for t in range(len(obs_ep)):
        buf.append(obs_ep[t])
        stacked = np.concatenate(list(buf)).astype(np.float32)
        rows.append(history_mask.apply(stacked, keep, K, base))
    return np.stack(rows)


def main() -> None:
    d = np.load(DATASET)
    obs, ep = d["obs"].astype(np.float32), d["ep_id"]
    eps = np.unique(ep)[:8]          # eight episodes is plenty to catch an offset
    ok = True
    for keep in ["all", "none", "scan", "self"]:
        worst = 0.0
        n = 0
        for e in eps:
            idx = np.where(ep == e)[0]
            o = obs[idx]
            train_rows = history_mask.apply(
                stack_frames(o, np.zeros(len(o), dtype=np.int32), K), keep, K, o.shape[1])
            eval_rows = eval_path(o, keep)
            worst = max(worst, float(np.abs(train_rows - eval_rows).max()))
            n += len(o)
        status = "identical" if worst == 0.0 else f"DIFFER by {worst:.3e}"
        ok &= worst == 0.0
        print(f"  keep={keep:5s} {n:6d} rows over {len(eps)} episodes: {status}")

    # The masking must actually do something, or a null result means nothing.
    print()
    o = obs[np.where(ep == eps[0])[0]]
    full = stack_frames(o, np.zeros(len(o), dtype=np.int32), K)
    for keep in ["none", "scan", "self"]:
        m = history_mask.apply(full, keep, K, o.shape[1])
        changed = int((np.abs(m - full) > 0).any(axis=0).sum())
        frozen = history_mask.frozen_indices(keep, o.shape[1]).size
        print(f"  keep={keep:5s} alters {changed:3d} of {full.shape[1]} input columns "
              f"({frozen} channels x {K-1} past frames = {frozen*(K-1)} expected)")

    # And the current frame must survive untouched in every setting.
    print()
    cur = slice((K - 1) * o.shape[1], K * o.shape[1])
    for keep in ["none", "scan", "self"]:
        m = history_mask.apply(full, keep, K, o.shape[1])
        same = np.array_equal(m[:, cur], full[:, cur])
        print(f"  keep={keep:5s} current frame preserved: {same}")
        ok &= same

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
