#!/usr/bin/env python3
"""Which part of the frame stack carries the stacking benefit?

    python3 scripts/history_tests.py

Five cells, all scored on the same 300 episodes of the 7x7 arena with overrides
off and no action repeat. The three ablation cells keep a 160-dim input and
differ only in which channel groups retain history across the four frames, so
they have identical capacity to the shipped 4-frame policy.

  1 frame     40-dim, no history at all       (the published 1-frame cell)
  none       160-dim, history frozen out      (capacity control for the above)
  scan       160-dim, lidar history only
  self       160-dim, goal distance/bearing history only
  4 frames   160-dim, full history            (the published 4-frame cell)

`none` is the zero point for the decomposition rather than the 1-frame cell,
because it isolates information from input size.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = {
    "1 frame": ["checkpoints/substrate_s5_dagger/bc_fs1_ar1.json",
                "checkpoints/bc_seeds/fs1_seed1.json",
                "checkpoints/bc_seeds/fs1_seed2.json",
                "checkpoints/bc_seeds/fs1_seed3.json"],
    "none":    [f"checkpoints/hist_ablation/none_seed{s}.json" for s in (0, 1, 2)],
    "scan":    [f"checkpoints/hist_ablation/scan_seed{s}.json" for s in (0, 1, 2)],
    "self":    [f"checkpoints/hist_ablation/self_seed{s}.json" for s in (0, 1, 2)],
    "4 frames": ["checkpoints/substrate_s5_dagger/bc_fs4_ar1.json",
                 "checkpoints/bc_seeds/fs4_seed1.json",
                 "checkpoints/bc_seeds/fs4_seed2.json",
                 "checkpoints/bc_seeds/fs4_seed3.json"],
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def load(files):
    out = []
    for f in files:
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            print(f"  missing: {f}")
            continue
        d = json.load(open(path))
        ep = d.get("per_episode")
        if ep:
            v = np.array([bool(e["success"]) for e in ep], dtype=bool)
        else:
            v = np.zeros(d["episodes"], dtype=bool)
            v[: d["successes"]] = True
        out.append(v)
    return out


def boot_diff(a, b, n_boot=20000, seed=0):
    """95% CI on mean(a) - mean(b), resampling seeds then episodes within seed."""
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ma = np.mean([rng.choice(a[j], a[j].size, replace=True).mean()
                      for j in rng.integers(0, len(a), len(a))])
        mb = np.mean([rng.choice(b[j], b[j].size, replace=True).mean()
                      for j in rng.integers(0, len(b), len(b))])
        d[i] = 100 * (ma - mb)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cells = {}
    print("Per-seed success on the same 300 episodes (7x7, no overrides, ar=1)\n")
    for name, files in CELLS.items():
        vecs = load(files)
        if not vecs:
            continue
        cells[name] = vecs
        rs = [100 * v.sum() / v.size for v in vecs]
        per = "  ".join(f"{r:5.1f}" for r in rs)
        sd = np.std(rs, ddof=1) if len(rs) > 1 else float("nan")
        print(f"  {name:9s} {per}   mean {np.mean(rs):5.1f}  sd {sd:4.1f}  "
              f"(n={len(rs)})")

    need = ("none", "scan", "self", "4 frames")
    if not all(k in cells for k in need):
        print("\nablation cells incomplete; rerun once the sweep finishes")
        return

    m = {k: float(np.mean([100 * v.sum() / v.size for v in cells[k]])) for k in cells}
    base, full = m["none"], m["4 frames"]
    span = full - base

    print("\n" + "-" * 66)
    print(f"no history {base:.1f}%  ->  full history {full:.1f}%   "
          f"span {span:+.1f} points\n")
    for k in ("scan", "self"):
        lo, hi = boot_diff(cells[k], cells["none"])
        share = 100 * (m[k] - base) / span if abs(span) > 1e-9 else float("nan")
        print(f"  {k:4s} history alone: {m[k]:5.1f}%  "
              f"{m[k]-base:+5.1f} vs no history  95% CI [{lo:+5.1f}, {hi:+5.1f}]  "
              f"recovers {share:5.1f}% of the span")
    print()
    for k in ("scan", "self"):
        lo, hi = boot_diff(cells["4 frames"], cells[k])
        print(f"  full minus {k:4s}:   {full-m[k]:+5.1f}  95% CI [{lo:+5.1f}, {hi:+5.1f}]")

    lo, hi = boot_diff(cells["none"], cells["1 frame"])
    print(f"\n  capacity control, no history minus 1 frame: "
          f"{base-m['1 frame']:+.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]")
    print("  (a gap here would mean input size, not history, moved the 1-frame cell)")

    # ── paired by seed ───────────────────────────────────────────────────────
    # A seed index is the same torch seed in every ablation cell: the same
    # weight init and the same train/val split, with only the masking changed.
    # The cells are therefore matched, and the spread BETWEEN seeds (seed 2
    # runs high everywhere, seed 1 low) is shared variance that pairing
    # removes. The unpaired intervals above carry it and are correspondingly
    # wide. Pairing over seeds is what Appendix I does for the frozen-obstacle
    # interaction, so it is this paper's standing unit, not a choice made here.
    print("\npaired by seed (same init and split, only the mask differs)")
    rates = {k: np.array([100 * v.sum() / v.size for v in cells[k]])
             for k in ("none", "scan", "self")}
    for k in ("scan", "self"):
        d = rates[k] - rates["none"]
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd / np.sqrt(d.size)
        t = mean / se if se > 0 else float("inf")
        # Exact sign-flip over the paired differences: 2^n assignments.
        n = d.size
        flips = [float(np.mean(d * np.array(sg))) for sg in
                 np.array(list(np.ndindex(*(2,) * n))) * 2 - 1]
        p_sf = float(np.mean(np.abs(flips) >= abs(mean) - 1e-12))
        per = "  ".join(f"{x:+5.1f}" for x in d)
        print(f"  {k:4s} minus none: {per}   mean {mean:+5.2f}  sd {sd:4.2f}  "
              f"t({n-1}) = {t:5.2f}  sign-flip p = {p_sf:.3f} (floor {2/2**n:.3f})")
    d = rates["scan"] - rates["self"]
    per = "  ".join(f"{x:+5.1f}" for x in d)
    print(f"  scan minus self: {per}   mean {float(d.mean()):+5.2f}  "
          f"(signs disagree)" if np.ptp(np.sign(d)) > 0 else "")

    print("\nreading:")
    sc, se = m["scan"] - base, m["self"] - base
    if span <= 0:
        print("  the span itself did not reproduce; nothing to decompose")
    elif sc > 0.6 * span and se < 0.3 * span:
        print("  lidar history carries it: obstacle motion relative to the robot")
    elif se > 0.6 * span and sc < 0.3 * span:
        print("  self-motion history carries it: knowing its own recent movement")
    elif sc > 0.3 * span and se > 0.3 * span:
        print("  both groups contribute")
    else:
        print("  neither group alone reproduces the span; read the intervals")


if __name__ == "__main__":
    main()
