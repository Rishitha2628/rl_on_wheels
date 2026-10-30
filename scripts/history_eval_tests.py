#!/usr/bin/env python3
"""What does a trained 4-frame policy rely on? Evaluation-time history ablation.

    python3 scripts/history_eval_tests.py

The same four checkpoints are scored unmodified and with part of their frame
stack's history overwritten by the current frame. Because the policy is
identical across cells, training noise cancels rather than being averaged over,
and episodes are paired within seed on the pinned goal sequence.

  self  past scans frozen, self-motion history intact
  scan  past self-motion frozen, scan history intact
  none  both frozen

A frozen stack is an input the policy never trained on, so a drop measures what
it relies on, not what it learned.
"""
from __future__ import annotations

import json
import math
import os
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF = ["checkpoints/substrate_s5_dagger/bc_fs4_ar1.json",
       "checkpoints/bc_seeds/fs4_seed1.json",
       "checkpoints/bc_seeds/fs4_seed2.json",
       "checkpoints/bc_seeds/fs4_seed3.json"]
CELLS = {c: [f"checkpoints/hist_eval/{c}_seed{s}.json" for s in range(4)]
         for c in ("self", "scan", "none")}


def vec(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    d = json.load(open(full))
    ep = d.get("per_episode")
    if ep:
        return np.array([bool(e["success"]) for e in ep], dtype=bool)
    v = np.zeros(d["episodes"], dtype=bool)
    v[: d["successes"]] = True
    return v


def mcnemar(a, b):
    """Exact two-sided binomial on the discordant pairs."""
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, p)


def main():
    ref = [vec(p) for p in REF]
    if any(v is None for v in ref):
        raise SystemExit("missing an unmodified reference run")
    rref = [100 * v.mean() for v in ref]
    print("unmodified 4-frame policies (same checkpoints, same episodes)")
    print("  " + "  ".join(f"{r:5.1f}" for r in rref) +
          f"   mean {np.mean(rref):5.1f}\n")

    rows = {}
    for cell, files in CELLS.items():
        vs = [vec(p) for p in files]
        if any(v is None for v in vs):
            print(f"  {cell}: incomplete")
            continue
        rows[cell] = vs

    if not rows:
        print("no eval-ablation results yet")
        return

    label = {"self": "scan history destroyed", "scan": "self-motion destroyed",
             "none": "all history destroyed"}
    print(f"{'cell':24s} {'per-seed delta vs own unmodified run':38s}  mean")
    for cell, vs in rows.items():
        deltas, mc = [], []
        for v, r in zip(vs, ref):
            deltas.append(100 * (v.mean() - r.mean()))
            mc.append(mcnemar(r, v))
        per = "  ".join(f"{d:+6.1f}" for d in deltas)
        print(f"{label[cell]:24s} {per:38s} {np.mean(deltas):+6.1f}")
        for s, (n10, n01, p) in enumerate(mc):
            print(f"    seed {s}: McNemar {n10} lost / {n01} gained, p = {p:.4f}")

    # Sign test over seeds, and whether the parts add to the whole.
    print()
    for cell, vs in rows.items():
        d = np.array([100 * (v.mean() - r.mean()) for v, r in zip(vs, ref)])
        neg = int((d < 0).sum())
        p = 2 * sum(math.comb(len(d), i) for i in range(min(neg, len(d) - neg) + 1)) / 2 ** len(d)
        print(f"  {label[cell]:24s} mean {d.mean():+5.1f}  "
              f"sd {d.std(ddof=1):4.1f}  {neg}/{len(d)} seeds down  "
              f"sign test p = {min(1.0, p):.3f}")

    if all(c in rows for c in ("self", "scan", "none")):
        def m(c):
            return float(np.mean([100 * (v.mean() - r.mean())
                                  for v, r in zip(rows[c], ref)]))
        a, b, t = m("self"), m("scan"), m("none")
        print(f"\n  parts {a:+.1f} (scan destroyed) and {b:+.1f} "
              f"(self-motion destroyed) sum to {a+b:+.1f}; "
              f"destroying both costs {t:+.1f}")
        if abs(a + b) > abs(t) + 1.0:
            print("  -> the two overlap: either alone accounts for more than its share")
        elif abs(a + b) < abs(t) - 1.0:
            print("  -> super-additive: the stack needs both together")
        else:
            print("  -> roughly additive")


if __name__ == "__main__":
    main()
