#!/usr/bin/env python3
"""Seed-level test of the control-rate null.

    python3 scripts/rate_seed_tests.py

Each arm is four separately trained policies, each scored on the same 300
reproducible episodes (goal seed 1000, 7x7 arena). The 2.5 Hz arm trains on
demonstrations resampled to 2.5 Hz and is evaluated with action_repeat=4; the
10 Hz arm trains on the full 10 Hz set and is evaluated with action_repeat=1.
Each policy therefore meets the frame spacing it trained on.

Two tests, because they answer different questions:

  * A hierarchical bootstrap resamples seeds with replacement and then episodes
    within each resampled seed, so the interval carries training noise and
    episode noise together. This is the interval to quote.
  * An exact permutation test over the eight seed-level rates asks whether the
    arm labels explain the spread at all. With 4 against 4 there are only
    C(8,4)=70 arrangements, so the smallest two-sided p it can return is
    2/70 = 0.029; a null here is a statement about power as much as about rates.

Episode-level pairing is deliberately not used. The two arms run different
policies over the same goal sequence, but a seed is the unit the claim is about.
"""
from __future__ import annotations

import json
import math
import os
from itertools import combinations

import numpy as np

ARMS = {
    "2.5 Hz": [
        "checkpoints/rate_matched/ratematched_fs4_2p5hz.json",
        "checkpoints/rate_seeds/ratematched_fs4_seed1.json",
        "checkpoints/rate_seeds/ratematched_fs4_seed2.json",
        "checkpoints/rate_seeds/ratematched_fs4_seed3.json",
    ],
    "10 Hz": [
        "checkpoints/substrate_s5_dagger/bc_fs4_ar1.json",
        "checkpoints/bc_seeds/fs4_seed1.json",
        "checkpoints/bc_seeds/fs4_seed2.json",
        "checkpoints/bc_seeds/fs4_seed3.json",
    ],
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load_arm(files: list[str]) -> list[np.ndarray]:
    """One boolean success vector per seed."""
    out = []
    for f in files:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            raise SystemExit(f"missing: {f}")
        d = json.load(open(path))
        ep = d.get("per_episode")
        if ep:
            v = np.array([bool(e["success"]) for e in ep], dtype=bool)
        else:  # fall back to the aggregate if per-episode was not stored
            v = np.zeros(d["episodes"], dtype=bool)
            v[: d["successes"]] = True
        out.append(v)
    return out


def hier_bootstrap(a: list[np.ndarray], b: list[np.ndarray],
                   n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ma = np.mean([rng.choice(a[j], a[j].size, replace=True).mean()
                      for j in rng.integers(0, len(a), len(a))])
        mb = np.mean([rng.choice(b[j], b[j].size, replace=True).mean()
                      for j in rng.integers(0, len(b), len(b))])
        diffs[i] = 100 * (ma - mb)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def exact_perm(ra: list[float], rb: list[float]) -> tuple[float, float, int]:
    """Two-sided exact permutation over seed-level rates. Returns (obs, p, n)."""
    allr = np.array(ra + rb, dtype=float)
    na = len(ra)
    obs = float(np.mean(ra) - np.mean(rb))
    stats = []
    for idx in combinations(range(allr.size), na):
        sel = np.zeros(allr.size, dtype=bool)
        sel[list(idx)] = True
        stats.append(allr[sel].mean() - allr[~sel].mean())
    stats = np.array(stats)
    p = float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    return obs, p, stats.size


def main() -> None:
    arms = {k: load_arm(v) for k, v in ARMS.items()}
    rates = {}
    for name, vecs in arms.items():
        print(f"\n{name}  (trained and evaluated at this rate)")
        rs = []
        for sd, v in enumerate(vecs):
            k, n = int(v.sum()), v.size
            lo, hi = wilson(k, n)
            rs.append(100 * k / n)
            print(f"  seed {sd}: {k:3d}/{n} = {100*k/n:5.1f}%  [{lo:4.1f},{hi:4.1f}]")
        m = float(np.mean(rs))
        s = float(np.std(rs, ddof=1))
        print(f"  mean {m:5.1f}%   sd across seeds {s:4.1f}")
        rates[name] = rs

    a, b = arms["2.5 Hz"], arms["10 Hz"]
    ra, rb = rates["2.5 Hz"], rates["10 Hz"]
    gap = float(np.mean(ra) - np.mean(rb))

    print("\n" + "-" * 62)
    print(f"2.5 Hz minus 10 Hz, mean over seeds: {gap:+.1f} points")
    lo, hi = hier_bootstrap(a, b)
    print(f"  hierarchical bootstrap 95% CI (seeds then episodes): [{lo:+.1f}, {hi:+.1f}]")
    obs, p, n_arr = exact_perm(ra, rb)
    print(f"  exact permutation over {len(ra)}+{len(rb)} seed rates: p = {p:.3f} "
          f"({n_arr} arrangements, floor {2/n_arr:.3f})")

    # What the single-seed comparison the paper currently quotes looked like.
    print(f"\n  for reference, seed 0 against seed 0: "
          f"{ra[0]:.1f} vs {rb[0]:.1f} = {ra[0]-rb[0]:+.1f}")
    print(f"  seed 0 is rank {sorted(rb, reverse=True).index(rb[0])+1} of "
          f"{len(rb)} in the 10 Hz arm")


if __name__ == "__main__":
    main()
