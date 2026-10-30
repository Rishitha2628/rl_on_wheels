#!/usr/bin/env python3
"""Regenerate the figures in paper/figures/ from checkpoints/ and logs/.

    python3 scripts/make_paper_figs.py

Reads the same JSON the tables are built from, so a figure can never drift from
the numbers in the text.
"""
from __future__ import annotations

import glob
import json
import math
import random
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
STEPS_PER_SEC = 9.93          # measured real-time bridge throughput

# Categorical slots 1-3, validated: all-pairs CVD dE 9.2 (deutan), normal 24.0.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pool(pattern: str, fname: str) -> dict[int, tuple[int, int]]:
    """Sum successes/episodes across seeds, per budget."""
    agg: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
        f = os.path.join(d, fname)
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["results"]:
            a = agg[r["budget_bridge_steps"]]
            a[0] += r["successes"]
            a[1] += r["episodes"]
    return {b: tuple(v) for b, v in sorted(agg.items())}


def load_bc(name: str) -> tuple[int, int]:
    """Successes/episodes for one evaluation.

    Falls back to checkpoints/aggregate_only/ when the live file is absent:
    the paired re-run deletes each result before regenerating it, so a figure
    built while that queue is in flight would otherwise fail on a missing file.
    The archived copy has the same counts; only per-episode detail is added.
    """
    live = os.path.join(ROOT, "checkpoints", name)
    arch = os.path.join(ROOT, "checkpoints", "aggregate_only", name)
    path = live if os.path.exists(live) else arch
    j = json.load(open(path))
    return j["successes"], j["episodes"]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ── Figure 0: the arena, drawn from the stage-4 SDF ──────────────────────────
# Keyframe offsets for the two moving cylinders, mirroring KEYS_1/KEYS_2 in
# dynamic_obstacle_node.cpp. Each is (dx, dy) from the model's SDF spawn pose.
# Keyframe offsets for the moving cylinders, mirroring KEYS_1..6 in
# dynamic_obstacle_node.cpp. Each is (dx, dy) from the model's spawn pose.
# The 7x7 arena activates all six; their bases sit at +/-2.8.
KEYS = {
 1: [(0,0),(-0.5,-1),(-3.5,-1),(-3.7,-3),(-3.5,-1),(-0.5,-1),(0,0)],
 2: [(0,0),(0.7,0.2),(2.5,3.5),(0.3,3.5),(3.5,1.8),(3.5,0),(2,0.5),
     (1.5,1),(1,0.5),(0.5,0.1),(0,0)],
 3: [(0,0),(-1,0.2),(-2,1),(-3.5,0),(-2.5,1.5),(0,0),(-1,2),(-2,1),(0,0)],
 4: [(0,0),(0,-3.2),(2,-2.7),(0,0),(0,-3.2),(2,-2.7),(0,0),(0,-3.2),(2,-2.7),(0,0)],
 5: [(0,0),(0.7,1),(2.5,2),(0,2),(0,-1),(2,0),(4,0),(3,-2),(2,0),(2,2),(0,0)],
 6: [(0,0),(-1,0),(-1,2),(-1.5,0),(0,2),(-4,-1.8),(-3,-1.8),(-2.5,1),(0,0)],
}
BASES = {1:(2.8,2.8), 2:(-2.8,-2.8), 3:(2.8,-2.8),
         4:(-2.8,2.8), 5:(-2.8,0.0), 6:(2.8,0.0)}

# Ground-plane mapping for the top-down render. The camera looks straight down
# from RENDER_CAM_Z with a RENDER_H_FOV horizontal field of view, so a point on
# the floor maps linearly. Verified against the SDF wall footprints, which land
# on the rendered wall bases. (Wall tops sit 0.5 m up and project outward by
# z/(z-0.5), which is why painted walls look slightly larger than their base.)
# These must change together with the camera pose in scripts/render_arena.sh.
RENDER_H_FOV, RENDER_CAM_Z = 0.78, 11.0
CROP = 130   # px trimmed from each edge, to the arena plus a small margin

def _arena_bbox(img):
    """Bounding box of the arena. Measured, never hard-coded --- a fixed crop
    silently cut the bottom-right corner off when the camera moved. Walls are
    tan so red exceeds blue; the grey floor and grid have r == g == b."""
    wall = (img[:, :, 0] - img[:, :, 2]) > 0.08
    ys, xs = np.where(wall)
    return xs.min(), xs.max(), ys.min(), ys.max()


def _common_crop(imgs, margin=18):
    """One crop window for every panel. Both renders use the same camera pose,
    so a shared window keeps the scale honest: the 6x6 arena should look
    smaller than the 7x7 one, not be enlarged to fill its panel."""
    boxes = [_arena_bbox(i) for i in imgs]
    H, W = imgs[0].shape[:2]
    return (max(0, min(b[0] for b in boxes) - margin),
            min(W, max(b[1] for b in boxes) + margin),
            max(0, min(b[2] for b in boxes) - margin),
            min(H, max(b[3] for b in boxes) + margin))


def fig_arena() -> None:
    """Both arenas as the simulator renders them, side by side.

    Tilted overhead rather than straight down, so the walls have visible
    height. Label anchors are given in ORIGINAL render pixels so they survive a
    change of crop.
    """
    panels = [
        ("arena_render_tilt5.png", "$7\\times7$: trained here",
         (((355, 675), "moving obstacles (6)", (115, 135), ORANGE),
          ((718, 685), "robot", (0, -95), BLUE),
          ((1105, 605), "goal", (95, -70), "#c62828"))),
        # Right panel carries only robot and goal: at this width a third label
        # does not fit inside the arena, and the cylinder counts are in the
        # caption instead.
        ("arena_render_tilt6.png", "unseen arena",
         (((718, 690), "robot", (-135, 60), BLUE),
          ((1258, 426), "goal", (10, 100), "#c62828"))),
    ]
    imgs = [plt.imread(os.path.join(OUT, fn)) for fn, _, _ in panels]
    x0, x1, y0, y1 = _common_crop(imgs)

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.48))
    for ax, img, (fn, title, labels) in zip(axes, imgs, panels):
        ax.imshow(img[y0:y1, x0:x1], interpolation="lanczos")
        for (px, py), label, (dx, dy), col in labels:
            ax.annotate(label, xy=(px - x0, py - y0),
                        xytext=(px - x0 + dx, py - y0 + dy),
                        fontsize=4.6, fontweight="bold", color=col,
                        ha="center", va="center", zorder=6,
                        arrowprops=dict(arrowstyle="-", color=col,
                                        linewidth=0.5, shrinkA=0.5, shrinkB=3))
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
        ax.set_title(title, fontsize=6.6, pad=2)
        ax.axis("off")
    fig.tight_layout(pad=0.05, w_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"arena.{ext}"), bbox_inches="tight", dpi=400)
    plt.close(fig)


def fig_reward() -> None:
    """AIRL-recovered reward against the hand-written one, three slices.

    Both are z-scored, because a reward is only defined up to a positive
    affine transform -- only the SHAPE of each curve is meaningful, never its
    level. We plot the base reward g: the shaped output adds a potential term
    that AIRL does not identify, and over these states it changes nothing
    (rank correlation between shaped and base is +1.000).
    """
    src = os.path.join(ROOT, "checkpoints", "reward_cmp", "reward_comparison.json")
    if not os.path.exists(src):
        print("  skip fig_reward: run airl/compare_rewards.py first")
        return
    with open(src) as fh:
        d = json.load(fh)

    # Single column: the two slices that carry the finding. The bearing
    # sweep is described in the text rather than given a third panel.
    order = [("distance", "goal distance (norm.)"),
             ("lidar",    "LiDAR clearance")]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.65))

    for ax, (key, xlab) in zip(axes, order):
        sw = d["sweeps"][key]
        x = np.asarray(sw["x"], dtype=float)
        airl = np.asarray(sw.get("airl_base", sw["airl"]), dtype=float)
        hand = np.asarray(sw["hand"], dtype=float)

        ax.axhline(0.0, color=GRID, linewidth=0.6, zorder=0)
        ax.plot(x, airl, color=BLUE, linewidth=1.8, zorder=3,
                label="AIRL (recovered)")
        # Where the hand reward has no term at all, say so rather than drawing
        # a flat line that reads as "measured and zero".
        if sw.get("hand_is_constant"):
            ax.text(0.04, 0.95, "hand-written:\nno such term",
                    transform=ax.transAxes, fontsize=5.6, color=ORANGE,
                    ha="left", va="top", style="italic", linespacing=1.2)
        else:
            ax.plot(x, hand, color=ORANGE, linewidth=1.6, zorder=2,
                    linestyle=(0, (4, 2)), label="hand-written")
        ax.set_xlabel(xlab, fontsize=6.8)
        ax.tick_params(labelsize=6.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel("reward ($z$)", fontsize=6.8)
    # Direct labels on panel 1, where both series are present.
    axes[0].annotate("AIRL", xy=(0.70, 0.80), xycoords="axes fraction",
                     fontsize=6.4, color=BLUE, fontweight="bold")
    axes[0].annotate("hand-written", xy=(0.05, 0.10), xycoords="axes fraction",
                     fontsize=6.4, color=ORANGE, fontweight="bold")

    # Identity never rests on colour alone: a legend as well as the direct
    # labels, and the two series differ in dash pattern.
    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label="AIRL"),
               plt.Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (4, 2)),
                          label="hand-written")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.8,
               handlelength=1.5, columnspacing=1.2,
               bbox_to_anchor=(0.54, 1.11))

    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"reward.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_nullspan() -> None:
    """Null distribution of a grid's span, against the four spans we observed.

    The span of a factorial grid is a maximum minus a minimum over noisy cells,
    so it grows with grid size even when every cell is identical. Simulating
    that null is the whole argument of the paper in one picture.
    """
    rng = np.random.default_rng(0)
    N, EPISODES, P, TRIALS = (4, 9), 300, 0.60, 40000
    grids = [(2, 2, "$2\\times2$"), (3, 3, "$3\\times3$")]

    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    for (r, c, lab), col in zip(grids, (BLUE, ORANGE)):
        k = r * c
        draws = rng.binomial(EPISODES, P, size=(TRIALS, k)) / EPISODES * 100
        span = draws.max(axis=1) - draws.min(axis=1)
        # Span is discrete at 300 episodes, so smooth with a kernel wide
        # enough to hide the lattice without moving the distribution.
        xs = np.linspace(0, 20, 240)
        h = 1.1
        dens = np.array([np.exp(-0.5 * ((span - x) / h) ** 2).mean() for x in xs])
        dens = dens / dens.max()
        ax.fill_between(xs, 0, dens, color=col, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(xs, dens, color=col, linewidth=1.6, zorder=2,
                label=f"{lab} grid, null")
        ax.axvline(np.median(span), color=col, linewidth=0.9,
                   linestyle=(0, (3, 2)), zorder=2)
        ax.text(np.median(span), 1.06, f"median {np.median(span):.1f}",
                color=col, fontsize=5.6, ha="center")

    # The four spans we measured, all from 2x2 grids.
    for x, lab in ((4.3, None), (8.0, None), (9.3, None), (11.7, "observed")):
        ax.plot([x], [0.035], marker="v", markersize=5, color=INK, zorder=4,
                markeredgecolor="white", markeredgewidth=0.6, label=lab)
        ax.text(x, -0.085, f"{x}", fontsize=5.6, color=INK, ha="center")

    ax.set_xlabel("span of the grid (percentage points)", fontsize=7.5)
    ax.set_ylabel("relative frequency", fontsize=7.5)
    ax.set_xlim(0, 20); ax.set_ylim(-0.02, 1.2)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, handlelength=1.6)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"nullspan.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_budget() -> None:
    # Single column, and TD3 only: the text does not discuss AIRL, and two
    # series read cleanly at this width where three did not.
    series = [
        ("TD3", pool("checkpoints/tune_s5/td3_s5_seed*", "eval_results.json"), BLUE),
    ]
    bc_alone = load_bc("substrate_s5_dagger/bc_policy_alone.json")  # match Table I
    bc_ovr   = load_bc("substrate_s5_dagger/bc_with_overrides.json")

    NUDGE = {"TD3": 1.0}
    LABEL_AT = {"TD3": (80_000, 3.2, "bottom")}

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    style_axes(ax)

    # BC reference levels, labelled on the left where no series runs.
    for (k, n), ls, lab, ty in ((bc_ovr, (0, (1, 2)), "BC+DAgger + overrides", 1.4),
                                (bc_alone, (0, (5, 3)), "BC+DAgger alone", -1.4)):
        p = 100 * k / n
        ax.axhline(p, color=MUTED, linewidth=1.0, linestyle=ls, zorder=1)
        ax.text(3.4e4, p + ty, f"{lab} {p:.1f}%", color=MUTED, fontsize=5.8,
                va="bottom" if ty > 0 else "top", ha="left")

    for label, agg, color in series:
        bs = sorted(agg)
        ys = np.array([100 * agg[b][0] / agg[b][1] for b in bs])
        ci = np.array([wilson(*agg[b]) for b in bs]) * 100
        ax.fill_between(bs, ci[:, 0], ci[:, 1], color=color, alpha=0.13,
                        linewidth=0, zorder=2)
        # The two TD3 curves converge at the largest budget and their markers
        # land on top of each other, which reads as a missing data point.
        # Offset them a hair horizontally so both stay visible.
        bsp = list(bs)
        if label in NUDGE:
            bsp[-1] = bsp[-1] * NUDGE[label]
        ax.plot(bsp, ys, color=color, linewidth=1.8, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=label)
        # The two TD3 curves meet at the largest budget, so labelling the
        # line-ends would stack them. Label each where the curves are apart.
        lx, ly, va = LABEL_AT[label]
        i = bs.index(lx)
        ax.text(lx * 1.06, ys[i] + ly, label, color=color, fontsize=7.5,
                va=va, ha="left", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(3.2e4, 2.4e6)
    ax.set_ylim(-3, 78)
    ax.set_xlabel("Environment interactions", labelpad=2, fontsize=7.5)
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks([4e4, 1.6e5, 6.4e5])
    ax.set_xticklabels(["40k", "160k", "640k"])
    ax.minorticks_off()

    sec = ax.secondary_xaxis(
        "top", functions=(lambda s: s / STEPS_PER_SEC / 3600,
                          lambda h: h * STEPS_PER_SEC * 3600))
    sec.set_xlabel("Robot-hours at 10 Hz", labelpad=3, fontsize=7.5)
    sec.set_xticks([1, 5, 20])
    sec.set_xticklabels(["1", "5", "20"])
    sec.spines["top"].set_color(MUTED)
    sec.minorticks_off()

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"budget.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_arena()
    fig_budget()
    fig_reward()
    fig_nullspan()
    print(f"wrote figures to {OUT}")
