#!/usr/bin/env python3
"""Every settled result, in one place, for writing the paper from.

    python3 scripts/summarize_all.py

Reads checkpoints/ directly so the paper can never drift from the runs.
Pre-fix results (checkpoints/prefix_*) are listed separately and must NOT be
mixed with the rest: they predate the /reset_goal_sequence fix and were scored
on different goal sets per cell.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPS = 9.93


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def diff_ci(a, b, na, nb):
    """95% half-width on the difference of two independent proportions (points)."""
    pa, pb = a / 100, b / 100
    return 1.96 * math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb) * 100


def one(path):
    f = os.path.join(ROOT, path)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    k, n = d["successes"], d["episodes"]
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, p=100 * k / n, lo=lo, hi=hi, len=d.get("mean_length"))


def show(label, path):
    r = one(path)
    if r:
        print(f"  {label:34s} {r['k']:3d}/{r['n']:<3d} = {r['p']:5.1f}%  [{r['lo']:4.1f},{r['hi']:4.1f}]")
    return r


def floor(pat, label):
    fs = sorted(glob.glob(os.path.join(ROOT, pat)))
    if not fs:
        return None
    s = [100 * json.load(open(f))["success_rate"] for f in fs]
    n = json.load(open(fs[0]))["episodes"]
    p = statistics.mean(s) / 100
    thr = 1.96 * math.sqrt(2) * 100 * math.sqrt(p * (1 - p) / 300)
    print(f"\n{label}")
    print(f"  repeats (n={n} each): {' '.join(f'{x:.0f}%' for x in s)}")
    print(f"  spread {max(s)-min(s):.1f} pts, sd {statistics.stdev(s):.1f}, mean {statistics.mean(s):.1f}%")
    print(f"  => threshold for a real difference at n=300: +/-{thr:.1f} pts")
    return thr


def arm(pat, label):
    agg, per = defaultdict(lambda: [0, 0]), defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(ROOT, pat))):
        j = json.load(open(f))
        s = j["meta"]["seed"]
        for r in j["results"]:
            b = r["budget_bridge_steps"]
            agg[b][0] += r["successes"]
            agg[b][1] += r["episodes"]
            per[b][s] = r["success_rate"]
    if not agg:
        return
    print(f"\n{label}")
    for b in sorted(agg):
        k, n = agg[b]
        lo, hi = wilson(k, n)
        seeds = " ".join(f"{100*per[b][s]:.0f}" for s in sorted(per[b]))
        print(f"  {b:>7} ({b/SPS/3600:4.1f} rh): {k:3d}/{n:<3d} = {100*k/n:5.1f}% [{lo:4.1f},{hi:4.1f}]  seeds: {seeds}")


print("=" * 72)
print("ALL POST-FIX RESULTS  (goal sequence rewound before every evaluation)")
print("=" * 72)

t4 = floor("checkpoints/floor/s4_*.json", "NOISE FLOOR - stage 4 policy")
t5 = floor("checkpoints/floor_s5/*.json", "NOISE FLOOR - stage 5 + DAgger policy")

print("\nSTAGE 4 - BC factorial (policy trained at 10 Hz, rate varied at eval)")
s4 = {c: show(c, f"checkpoints/substrate/{c}.json") for c in
      ("bc_fs1_ar1", "bc_fs1_ar4", "bc_fs4_ar1", "bc_fs4_ar4",
       "bc_policy_alone", "bc_with_overrides")}

print("\nSTAGE 5 + DAgger - BC factorial")
s5 = {c: show(c, f"checkpoints/substrate_s5_dagger/{c}.json") for c in
      ("bc_fs1_ar1", "bc_fs1_ar4", "bc_fs4_ar1", "bc_fs4_ar4",
       "bc_policy_alone", "bc_with_overrides")}

print("\nSTAGE 5 - BC before DAgger (undertrained; the 'collapse' that wasn't)")
s5n = {c: show(c, f"checkpoints/substrate_s5_nodagger/{c}.json") for c in
       ("bc_fs1_ar1", "bc_fs1_ar4", "bc_fs4_ar1", "bc_fs4_ar4",
        "bc_policy_alone", "bc_with_overrides")}

print("\nSTAGE 5 - rate-matched cells (trained AND evaluated at the same rate)")
rm = {c: show(c, f"checkpoints/rate_matched/{c}.json") for c in
      ("ratematched_fs1_2p5hz", "ratematched_fs4_2p5hz", "volctrl_fs4_10hz")}

arm("checkpoints/tune_s5/td3_s5_seed*/eval_results.json", "TD3 stage-5 TUNED (lr 3e-4, sigma 0.1)")
arm("checkpoints/tune_s5/td3_s5_shipped_seed*/eval_results.json", "TD3 stage-5 SHIPPED (lr 1e-4, sigma 0.05)")
arm("checkpoints/sweep_s5/airl_s5_seed*/eval_results.json", "AIRL stage-5")

print("\n" + "=" * 72)
print("KEY COMPARISONS vs the noise floor")
print("=" * 72)
for lbl, d, thr in (("stage 4", s4, t4), ("stage 5 + DAgger", s5, t5)):
    cells = [d[c] for c in ("bc_fs1_ar1", "bc_fs1_ar4", "bc_fs4_ar1", "bc_fs4_ar4") if d.get(c)]
    if len(cells) == 4 and thr:
        ps = [c["p"] for c in cells]
        span = max(ps) - min(ps)
        print(f"\n{lbl}: factorial span {span:.1f} pts  (threshold +/-{thr:.1f})  "
              f"{'CLEARS' if span > thr else 'INSIDE NOISE'}")
        print(f"   fs=1: {ps[0]:.1f} -> {ps[1]:.1f} ({ps[1]-ps[0]:+.1f})   "
              f"fs=4: {ps[2]:.1f} -> {ps[3]:.1f} ({ps[3]-ps[2]:+.1f})   "
              f"interaction {(ps[3]-ps[2])-(ps[1]-ps[0]):+.1f} pts")
    a, b = d.get("bc_policy_alone"), d.get("bc_with_overrides")
    if a and b and thr:
        dd = b["p"] - a["p"]
        print(f"   wrapper: {a['p']:.1f} -> {b['p']:.1f} = {dd:+.1f} pts "
              f"(+/-{diff_ci(a['p'], b['p'], a['n'], b['n']):.1f})  "
              f"{'CLEARS' if abs(dd) > thr else 'inside noise'}")
