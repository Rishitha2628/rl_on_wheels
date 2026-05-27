#!/usr/bin/env python3
"""Generate Nav2 occupancy grids (PGM + YAML) for each TurtleBot3 stage.

Maps include STATIC obstacles only — outer walls (all stages), inner walls
(stages 4, 5, 7-10), and static cylinders (stage 2). Moving obstacles are
handled by Nav2's local costmap at runtime via the live LiDAR.

Output: ros2_ws/src/tb3_nav2/maps/tb3_stage{N}.{pgm,yaml}
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── map specs ─────────────────────────────────────────────────────────────────
RESOLUTION = 0.05                              # m per pixel — Nav2 default
HALF_EXTENT = 2.8                              # half-side of map (1m margin past walls)
SIZE_PX = int(2 * HALF_EXTENT / RESOLUTION)    # 112x112 pixels

OCCUPIED_VAL = 0     # PGM: 0 = black = occupied
FREE_VAL     = 254   # PGM: 254 = white = free
INFLATE = 0.0        # leave the static map "raw"; Nav2's costmap inflates

OUT_DIR = Path(__file__).resolve().parent.parent / "maps"


# ── geometry (must match gen_stages.py) ───────────────────────────────────────
@dataclass
class WallRect:
    cx: float
    cy: float
    size_x: float  # length along wall's local x-axis
    size_y: float  # thickness along wall's local y-axis
    yaw: float


# Outer arena walls (5x5, walls at ±2.425, length 5m, thickness 0.15m)
OUTER_WALLS = [
    WallRect(-2.425, 0.0,    5.0, 0.15,  math.pi / 2),
    WallRect( 2.425, 0.0,    5.0, 0.15, -math.pi / 2),
    WallRect( 0.0,   2.425,  5.0, 0.15,  0.0),
    WallRect( 0.0,  -2.425,  5.0, 0.15,  math.pi),
]

# Inner walls (7 segments, length 1m, thickness 0.15m)
INNER_WALLS = [
    WallRect(-2.0,  -1.5,    1.0, 0.15,  0.0),
    WallRect(-0.5,  -2.0,    1.0, 0.15, -math.pi / 2),
    WallRect( 1.0,  -1.0,    1.0, 0.15,  math.pi / 2),
    WallRect( 1.2,   1.9,    1.0, 0.15, -math.pi / 2),
    WallRect( 1.9,   0.4,    1.0, 0.15,  0.0),
    WallRect(-0.5,   1.5,    1.0, 0.15,  0.0),
    WallRect(-1.2,   0.092,  1.0, 0.15, -math.pi / 2),
]

STATIC_CYLINDERS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
CYLINDER_RADIUS = 0.15

# Per-stage flags
HAS_INNER = {1: False, 2: False, 3: False, 4: True, 5: True,
             6: False, 7: True, 8: True, 9: True, 10: True}
HAS_STATIC_CYL = {1: False, 2: True, 3: False, 4: False, 5: False,
                  6: False, 7: False, 8: False, 9: False, 10: False}


# ── rasterization ─────────────────────────────────────────────────────────────
def world_to_px(x: float, y: float) -> tuple[int, int]:
    """Convert world coords to pixel coords.

    Image origin is at the bottom-left of the image (Nav2 convention).
    World (-HALF_EXTENT, -HALF_EXTENT) maps to image (0, SIZE_PX-1).
    """
    px = int((x + HALF_EXTENT) / RESOLUTION)
    # PGM row 0 is the TOP of the image, but the YAML origin tells Nav2
    # that the bottom-left pixel is at world origin. So we flip y.
    py = (SIZE_PX - 1) - int((y + HALF_EXTENT) / RESOLUTION)
    return px, py


def draw_rect(grid: np.ndarray, rect: WallRect) -> None:
    """Rasterize a rotated rectangle into the grid as occupied."""
    cos_y = math.cos(rect.yaw)
    sin_y = math.sin(rect.yaw)
    hx = rect.size_x / 2 + INFLATE
    hy = rect.size_y / 2 + INFLATE

    # Bounding box of the rotated rect → iterate over those pixels and test
    max_r = math.hypot(hx, hy)
    x0, x1 = rect.cx - max_r, rect.cx + max_r
    y0, y1 = rect.cy - max_r, rect.cy + max_r
    px0, py0 = world_to_px(x0, y1)
    px1, py1 = world_to_px(x1, y0)
    px_lo, px_hi = min(px0, px1), max(px0, px1)
    py_lo, py_hi = min(py0, py1), max(py0, py1)

    for py in range(max(0, py_lo), min(SIZE_PX, py_hi + 1)):
        for px in range(max(0, px_lo), min(SIZE_PX, px_hi + 1)):
            # pixel center back to world
            wx = (px + 0.5) * RESOLUTION - HALF_EXTENT
            wy = -((py + 0.5) * RESOLUTION - HALF_EXTENT + RESOLUTION) + HALF_EXTENT
            # Reverse the same y-flip done in world_to_px
            wy = HALF_EXTENT - (py + 0.5) * RESOLUTION
            # Rotate into rect-local frame
            dx, dy = wx - rect.cx, wy - rect.cy
            lx =  cos_y * dx + sin_y * dy
            ly = -sin_y * dx + cos_y * dy
            if abs(lx) <= hx and abs(ly) <= hy:
                grid[py, px] = OCCUPIED_VAL


def draw_circle(grid: np.ndarray, cx: float, cy: float, r: float) -> None:
    rr = r + INFLATE
    px_lo, py_hi = world_to_px(cx - rr, cy - rr)
    px_hi, py_lo = world_to_px(cx + rr, cy + rr)
    for py in range(max(0, py_lo), min(SIZE_PX, py_hi + 1)):
        for px in range(max(0, px_lo), min(SIZE_PX, px_hi + 1)):
            wx = (px + 0.5) * RESOLUTION - HALF_EXTENT
            wy = HALF_EXTENT - (py + 0.5) * RESOLUTION
            if math.hypot(wx - cx, wy - cy) <= rr:
                grid[py, px] = OCCUPIED_VAL


def build_stage_map(stage: int) -> np.ndarray:
    grid = np.full((SIZE_PX, SIZE_PX), FREE_VAL, dtype=np.uint8)
    for w in OUTER_WALLS:
        draw_rect(grid, w)
    if HAS_INNER[stage]:
        for w in INNER_WALLS:
            draw_rect(grid, w)
    if HAS_STATIC_CYL[stage]:
        for (cx, cy) in STATIC_CYLINDERS:
            draw_circle(grid, cx, cy, CYLINDER_RADIUS)
    return grid


# ── PGM + YAML output ─────────────────────────────────────────────────────────
def write_pgm(path: Path, grid: np.ndarray) -> None:
    h, w = grid.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(grid.tobytes())


def write_yaml(path: Path, pgm_name: str) -> None:
    """Nav2 map YAML — origin is the world coord of the bottom-left pixel."""
    yaml_text = (
        f"image: {pgm_name}\n"
        f"resolution: {RESOLUTION}\n"
        f"origin: [{-HALF_EXTENT}, {-HALF_EXTENT}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    path.write_text(yaml_text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stage in range(1, 11):
        grid = build_stage_map(stage)
        pgm_name = f"tb3_stage{stage}.pgm"
        write_pgm(OUT_DIR / pgm_name, grid)
        write_yaml(OUT_DIR / f"tb3_stage{stage}.yaml", pgm_name)
        n_occ = int((grid == OCCUPIED_VAL).sum())
        print(f"  stage {stage:2d}: {SIZE_PX}x{SIZE_PX} px, "
              f"{n_occ:5d} occupied  ({pgm_name})")
    print(f"Done. Maps in {OUT_DIR}")


if __name__ == "__main__":
    main()
