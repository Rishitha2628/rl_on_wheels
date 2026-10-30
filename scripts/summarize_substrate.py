#!/usr/bin/env python3
"""Print the substrate tables: the BC factorial and the action-repeat comparison.

Everything here is scored on the same fixed goal sequence, so the cells are
paired and differences between them are not partly a difference in which goals
each run happened to draw.
"""
import json, math, os, glob

def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, c - m), min(1.0, c + m)

OUT = "/checkpoints/substrate"

print("\nBC factorial, policy alone, fixed goals")
print("  %-12s %-14s %9s %16s %8s" % ("frame_stack", "action_repeat", "success", "95% CI", "episodes"))
print("  " + "-" * 64)
for fs in (1, 4):
    for ar in (1, 4):
        f = f"{OUT}/bc_fs{fs}_ar{ar}.json"
        if not os.path.exists(f):
            print("  %-12s %-14s %9s" % (fs, ar, "--"))
            continue
        d = json.load(open(f))
        lo, hi = wilson(d["successes"], d["episodes"])
        rate = "%.1f%% (%s Hz)" % (d["success_rate"] * 100, "10" if ar == 1 else "2.5")
        print("  %-12s %-14s %9s %16s %8d" % (
            fs, ar, rate, "[%.1f, %.1f]" % (lo * 100, hi * 100), d["episodes"]))

def pooled(paths, budget=None):
    s = n = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["results"]:
            if budget and r["budget_bridge_steps"] != budget:
                continue
            s += r["successes"]; n += r["episodes"]
    return s, n

print("\nTD3 control rate at 640k bridge steps, 3 seeds pooled")
print("  %-26s %9s %16s %8s" % ("config", "success", "95% CI", "episodes"))
print("  " + "-" * 64)
rows = [
    ("action_repeat 4 (2.5 Hz)", sorted(glob.glob("/checkpoints/tune/td3_confirm_seed*/eval_paired.json"))),
    ("action_repeat 1 (10 Hz)",  sorted(glob.glob(f"{OUT}/td3_ar1_seed*/eval_results.json"))),
]
for label, paths in rows:
    s, n = pooled(paths, budget=640000)
    if n == 0:
        print("  %-26s %9s" % (label, "--")); continue
    lo, hi = wilson(s, n)
    print("  %-26s %8.1f%% %16s %8d" % (
        label, s / n * 100, "[%.1f, %.1f]" % (lo * 100, hi * 100), n))

print("\n  For reference, unpaired: BC alone 63.7%, BC + overrides 78.0%,")
print("  TD3 tuned 68.0%, TD3 default 62.7%.\n")
