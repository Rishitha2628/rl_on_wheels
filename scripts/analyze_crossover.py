#!/usr/bin/env python3
"""Aggregate sweep evaluations into the paper's crossover figure and number.

Reads eval_results.json from each run directory, pools across seeds, and
answers the paper's question: at what interaction budget does online RL
overtake imitation, and what does that budget cost in robot-hours?

Two crossing definitions are reported, because they say different things:

  point crossing    first budget where the mean success rate exceeds BC's.
                    Easy to over-read -- with 50 episodes a 4-point gap is noise.

  separated crossing first budget where the method's 95% lower bound clears
                    BC's 95% upper bound. This is the defensible claim and the
                    one that belongs in the abstract.

Robot-hours convert at the real-time bridge rate (10 Hz measured at 9.93
steps/s), which is what a physical TurtleBot3 allows. That conversion is the
entire point of the exercise, so it is applied to the axis rather than left
for the reader.

Usage:
    python3 scripts/analyze_crossover.py --sweep-dir /checkpoints/sweep \
        --out-dir /logs/figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Categorical slots 1-3 of the validated palette. Fixed order, never cycled.
# All-pairs validated in light mode: worst CVD dE 9.2, worst normal-vision 24.0.
SERIES = {
    "bc":   {"color": "#2a78d6", "label": "BC + DAgger",   "ls": "--", "marker": "s"},
    "airl": {"color": "#eb6834", "label": "AIRL (learned reward)", "ls": "-.", "marker": "^"},
    "td3":  {"color": "#1baf7a", "label": "TD3 (task reward)", "ls": "-",  "marker": "o"},
    "td3_tuned": {"color": "#1baf7a", "label": "TD3 (tuned)", "ls": ":", "marker": "D"},
}

INK        = "#1a1a19"
INK_MUTED  = "#6b6a65"
GRID       = "#e2e2de"
SURFACE    = "#fcfcfb"

# Measured real-time bridge throughput (scripts/bench_throughput.py).
REALTIME_STEPS_PER_SEC = 9.93


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_method(sweep_dir: Path, prefix: str) -> dict:
    """Pool every seed's per-budget episode counts for one method.

    Pools raw successes/episodes rather than averaging per-seed rates: the
    quantity is binomial, and pooling keeps the interval honest when seeds
    contribute unequal episode counts (an interrupted run does).
    """
    runs = sorted(sweep_dir.glob(f"{prefix}_seed*"))
    pooled: dict[int, dict] = {}
    per_seed: dict[int, list] = {}

    for run in runs:
        res_path = run / "eval_results.json"
        if not res_path.exists():
            continue
        blob = json.loads(res_path.read_text())
        for r in blob.get("results", []):
            b = int(r["budget_bridge_steps"])
            slot = pooled.setdefault(b, {"successes": 0, "episodes": 0, "rewards": []})
            slot["successes"] += int(r["successes"])
            slot["episodes"] += int(r["episodes"])
            slot["rewards"].append(float(r["mean_reward"]))
            per_seed.setdefault(b, []).append(float(r["success_rate"]))

    points = []
    for b in sorted(pooled):
        s, n = pooled[b]["successes"], pooled[b]["episodes"]
        lo, hi = wilson(s, n)
        points.append({
            "budget": b,
            "rate": s / n if n else 0.0,
            "lo": lo, "hi": hi,
            "episodes": n, "successes": s,
            "n_seeds": len(per_seed[b]),
            "seed_rates": per_seed[b],
            "mean_reward": float(np.mean(pooled[b]["rewards"])),
        })
    return {"points": points, "n_runs": len(runs)}


def find_crossings(method: dict, bc_rate: float,
                   bc_hi: float) -> dict:
    point_x = sep_x = None
    for p in method["points"]:
        if point_x is None and p["rate"] > bc_rate:
            point_x = p["budget"]
        if sep_x is None and p["lo"] > bc_hi:
            sep_x = p["budget"]
    return {
        "point_crossing_bridge_steps": point_x,
        "separated_crossing_bridge_steps": sep_x,
        "point_crossing_robot_hours": (
            round(point_x / REALTIME_STEPS_PER_SEC / 3600, 1) if point_x else None),
        "separated_crossing_robot_hours": (
            round(sep_x / REALTIME_STEPS_PER_SEC / 3600, 1) if sep_x else None),
    }


def make_figure(methods: dict, bc: dict | None, out_dir: Path,
                bc_ovr: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    max_x = 0.0
    for m in methods.values():
        for p in m.get("points", []):
            max_x = max(max_x, float(p["budget"]))

    # BC is budget-independent -- it trains offline on a fixed demo set, so it
    # is a horizontal reference, not a curve. Drawing it as a line across the
    # budget axis is the whole visual point: the question is where the others
    # cross it.
    if bc:
        spec = SERIES["bc"]
        ax.axhline(bc["rate"], color=spec["color"], ls=spec["ls"], lw=1.6,
                   zorder=3, label=spec["label"])
        ax.axhspan(bc["lo"], bc["hi"], color=spec["color"], alpha=0.10,
                   lw=0, zorder=1)

    # The same learned policy wrapped in its hand-written safety guards and
    # near-goal P-controller. Same hue (same policy), solid (the added
    # control). This is the line every learned curve fails to reach, so
    # leaving it off the figure would omit the main result.
    if bc_ovr:
        ax.axhline(bc_ovr["rate"], color=SERIES["bc"]["color"], ls="-", lw=1.8,
                   zorder=3, label="BC + hand-written control")
        ax.axhspan(bc_ovr["lo"], bc_ovr["hi"], color=SERIES["bc"]["color"],
                   alpha=0.10, lw=0, zorder=1)
        ax.annotate("BC + hand-written control", xy=(max_x, bc_ovr["rate"]),
                    xytext=(-4, 5), textcoords="offset points",
                    color=SERIES["bc"]["color"], fontsize=7.5,
                    ha="right", va="bottom", fontweight="medium")

    for key in ("airl", "td3", "td3_tuned"):
        m = methods.get(key)
        if not m or not m["points"]:
            continue
        spec = SERIES[key]
        xs = np.array([p["budget"] for p in m["points"]], dtype=float)
        ys = np.array([p["rate"] for p in m["points"]])
        lo = np.array([p["lo"] for p in m["points"]])
        hi = np.array([p["hi"] for p in m["points"]])

        ax.fill_between(xs, lo, hi, color=spec["color"], alpha=0.14,
                        lw=0, zorder=2)
        ax.plot(xs, ys, color=spec["color"], ls=spec["ls"], lw=2.0,
                marker=spec["marker"], markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=spec["label"], zorder=4)

        # Per-seed points, so seed spread is visible rather than hidden inside
        # a pooled interval that only shows sampling error.
        for p in m["points"]:
            if len(p["seed_rates"]) > 1:
                ax.scatter([p["budget"]] * len(p["seed_rates"]), p["seed_rates"],
                           s=5, color=spec["color"], alpha=0.35,
                           linewidths=0, zorder=3)

        # Direct label at the line end. Required here, not decorative: the
        # palette validator flags one series below 3:1 on a light surface, and
        # direct labels are the relief the rule asks for. Anchored inside the
        # axes -- an offset label past the last point escapes the plot area and
        # clips once the figure is placed in a LaTeX column.
        ax.annotate(spec["label"], xy=(xs[-1], ys[-1]),
                    xytext=(6, 7), textcoords="offset points",
                    color=spec["color"], fontsize=7.5, va="bottom",
                    ha="left", fontweight="medium",
                    annotation_clip=False)

    ax.set_xscale("log")
    # Headroom on the right so the end-of-line labels sit inside the axes.
    if max_x > 0:
        ax.set_xlim(right=max_x * 3.2)
    ax.set_xlabel("Interaction budget (environment steps)", color=INK)
    ax.set_ylabel("Success rate (deterministic eval)", color=INK)
    ax.set_ylim(0, 1)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Secondary x-axis in robot-hours. This is a unit conversion of the same
    # axis, not a second measure on a second scale -- it restates the budget in
    # the currency the paper actually argues in.
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda s: s / REALTIME_STEPS_PER_SEC / 3600,
                   lambda h: h * REALTIME_STEPS_PER_SEC * 3600))
    sec.set_xlabel(f"Continuous real-robot operation (hours at "
                   f"{REALTIME_STEPS_PER_SEC:.1f} steps/s)", color=INK_MUTED,
                   fontsize=8)
    sec.tick_params(colors=INK_MUTED, labelsize=8)

    ax.legend(frameon=False, loc="upper left", fontsize=8,
              labelcolor=INK, handlelength=2.4)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"crossover.{ext}", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] wrote {out_dir}/crossover.png and .pdf")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", default="/checkpoints/sweep")
    p.add_argument("--bc-results", default=None,
                   help="JSON with BC's successes/episodes (budget-independent)")
    p.add_argument("--out-dir", default="/logs/figures")
    p.add_argument("--bc-overrides-results", default=None,
                   help="JSON for BC WITH its hand-written overrides; drawn as "
                        "a second reference line")
    p.add_argument("--tuned-dir", default="/checkpoints/tune",
                   help="directory holding td3_confirm_seed* (the tuned TD3 arm)")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR sweep dir not found: {sweep_dir}", file=sys.stderr)
        return 1

    methods = {k: load_method(sweep_dir, k) for k in ("td3", "airl")}

    # The tuned TD3 runs live outside the sweep directory, under checkpoints/
    # tune/td3_confirm_seed*. They matter: the in-sweep td3_seed* runs use the
    # configuration whose own comments say it was meant for fine-tuning an
    # already-converged policy, so plotting only those would show the arm at a
    # handicap we already know about and corrected.
    tuned_dir = Path(args.tuned_dir) if args.tuned_dir else None
    if tuned_dir and tuned_dir.exists():
        methods["td3_tuned"] = load_method(tuned_dir, "td3_confirm")
    else:
        methods["td3_tuned"] = {"points": [], "n_runs": 0}

    def _load_bc(path):
        if not path or not Path(path).exists():
            return None
        blob = json.loads(Path(path).read_text())
        s, n = int(blob["successes"]), int(blob["episodes"])
        lo, hi = wilson(s, n)
        return {"rate": s / n, "lo": lo, "hi": hi, "episodes": n, "successes": s}

    bc = _load_bc(args.bc_results)
    bc_ovr = _load_bc(args.bc_overrides_results)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n  {'method':<8} {'budget':>10} {'robot-h':>9} "
          f"{'success':>9} {'95% CI':>16} {'seeds':>6}")
    print("  " + "-" * 64)
    for key in ("td3", "td3_tuned", "airl"):
        for pt in methods[key]["points"]:
            rh = pt["budget"] / REALTIME_STEPS_PER_SEC / 3600
            ci = f"[{pt['lo']:.0%}, {pt['hi']:.0%}]"
            print(f"  {key:<8} {pt['budget']:>10,} {rh:>9.1f} "
                  f"{pt['rate']:>8.1%} {ci:>16} {pt['n_seeds']:>6}")
    if bc:
        bc_ci = f"[{bc['lo']:.0%}, {bc['hi']:.0%}]"
        print(f"  {'bc':<8} {'(offline)':>10} {'--':>9} "
              f"{bc['rate']:>8.1%} {bc_ci:>16} {'--':>6}")

    if bc_ovr:
        print("  %-8s %10s %9s %8.1f%% %16s" % (
            "bc+ovr", "(offline)", "--", bc_ovr["rate"] * 100,
            "[%.0f%%, %.0f%%]" % (bc_ovr["lo"] * 100, bc_ovr["hi"] * 100)))

    summary = {
        "realtime_steps_per_sec": REALTIME_STEPS_PER_SEC,
        "bc": bc,
        "bc_with_overrides": bc_ovr,
        "methods": {k: methods[k]["points"] for k in methods},
    }
    if bc:
        summary["crossings"] = {
            k: find_crossings(methods[k], bc["rate"], bc["hi"])
            for k in ("td3", "td3_tuned", "airl") if methods[k]["points"]
        }
        print("\n  crossings vs BC:")
        for k, c in summary["crossings"].items():
            sep = c["separated_crossing_robot_hours"]
            pt = c["point_crossing_robot_hours"]
            print(f"    {k}: point={pt if pt else 'never'} h, "
                  f"CI-separated={sep if sep else 'never'} h")
    else:
        print("\n  (no BC baseline supplied -- pass --bc-results to get crossings)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crossover.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[analyze] wrote {out_dir}/crossover.json")

    if not args.no_figure and any(methods[k]["points"] for k in methods):
        make_figure(methods, bc, out_dir, bc_ovr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
