#!/usr/bin/env python3
"""McNemar tests over the per-episode outcomes bc/eval.py now records.

    python3 scripts/paired_tests.py

Every evaluation rewinds the goal sequence first, so episode i is the same goal
in every run. That makes two runs comparable episode by episode: McNemar's test
looks only at the episodes where they DISAGREE (one succeeds, the other fails)
and ignores the rest, which is where its extra power over comparing two totals
comes from. Runs that predate the per-episode logging are skipped with a note
rather than silently compared as unpaired.
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    f = os.path.join(ROOT, "checkpoints", path + ".json")
    if not os.path.exists(f):
        return None, "missing"
    d = json.load(open(f))
    pe = d.get("per_episode")
    if not pe:
        return None, "aggregate-only (re-run needed)"
    return [bool(e["success"]) for e in pe], None


def mcnemar(a, b):
    """Exact binomial McNemar on the discordant pairs."""
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)      # a fails, b wins
    n10 = sum(1 for x, y in zip(a, b) if x and (not y))      # a wins, b fails
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n10, n01, min(1.0, p)


def compare(la, pa, lb, pb):
    a, ea = load(pa)
    b, eb = load(pb)
    if ea or eb:
        print(f"  {la} vs {lb}: skipped ({ea or eb})")
        return
    if len(a) != len(b):
        print(f"  {la} vs {lb}: skipped (lengths {len(a)} vs {len(b)})")
        return
    ra, rb = 100 * sum(a) / len(a), 100 * sum(b) / len(b)
    n10, n01, p = mcnemar(a, b)
    both = sum(1 for x, y in zip(a, b) if x and y)
    print(f"  {la:30s} {ra:5.1f}%   {lb:30s} {rb:5.1f}%")
    print(f"      agree-success {both:3d}   {la.split()[0]}-only {n10:3d}   "
          f"{lb.split()[0]}-only {n01:3d}   McNemar p = {p:.4f}"
          f"   {'significant' if p < 0.05 else 'n.s.'}")


PAIRS = [
    ("stage 5: the factorial", [
        ("fs4@10Hz", "substrate_s5_dagger/bc_fs4_ar1", "fs4@2.5Hz", "substrate_s5_dagger/bc_fs4_ar4"),
        ("fs1@10Hz", "substrate_s5_dagger/bc_fs1_ar1", "fs1@2.5Hz", "substrate_s5_dagger/bc_fs1_ar4"),
        ("fs4@10Hz", "substrate_s5_dagger/bc_fs4_ar1", "fs1@10Hz", "substrate_s5_dagger/bc_fs1_ar1"),
    ]),
    ("stage 5: the evaluation wrapper", [
        ("with overrides", "substrate_s5_dagger/bc_with_overrides",
         "policy alone", "substrate_s5_dagger/bc_policy_alone"),
    ]),
    ("stage 5: rate matching", [
        ("matched 2.5Hz", "rate_matched/ratematched_fs4_2p5hz",
         "mismatched 2.5Hz", "substrate_s5_dagger/bc_fs4_ar4"),
        ("matched 2.5Hz", "rate_matched/ratematched_fs4_2p5hz",
         "native 10Hz", "substrate_s5_dagger/bc_fs4_ar1"),
        ("matched 2.5Hz", "rate_matched/ratematched_fs4_2p5hz",
         "volume-ctrl 10Hz", "rate_matched/volctrl_fs4_10hz"),
    ]),
    ("stage 4: the factorial", [
        ("fs4@10Hz", "substrate/bc_fs4_ar1", "fs4@2.5Hz", "substrate/bc_fs4_ar4"),
        ("with overrides", "substrate/bc_with_overrides", "policy alone", "substrate/bc_policy_alone"),
    ]),
]

if __name__ == "__main__":
    for title, pairs in PAIRS:
        print(f"\n=== {title} ===")
        for la, pa, lb, pb in pairs:
            compare(la, pa, lb, pb)
