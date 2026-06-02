#!/usr/bin/env python3
"""Generate Nav2 occupancy grids (PGM + YAML) for each TurtleBot3 stage.

Maps include STATIC obstacles only — outer walls (all stages), inner walls
(stages 4 and 5), and static cylinders (stage 2). Moving obstacles are
handled by Nav2's local costmap at runtime via the live LiDAR.

Output: ros2_ws/src/tb3_nav2/maps/tb3_stage{N}.{pgm,yaml}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── map specs ─────────────────────────────────────────────────────────────────
RESOLUTION = 0.05                              # m per pixel — Nav2 default

OCCUPIED_VAL = 0     # PGM: 0 = black = occupied
FREE_VAL     = 254   # PGM: 254 = white = free
# Static safety margin baked into the map. Nav2 inflates further at runtime.
# 0.08m extra on each side of every wall = walls are 0.16m thicker in the map.
INFLATE = 0.08

OUT_DIR = Path(__file__).resolve().parent.parent / "maps"


# ── geometry ──────────────────────────────────────────────────────────────────
@dataclass
class WallRect:
    cx: float
    cy: float
    size_x: float  # length along wall's local x-axis
    size_y: float  # thickness along wall's local y-axis
    yaw: float


# Outer arena walls (5x5, walls at ±2.425, length 5m, thickness 0.15m)
OUTER_WALLS_5x5 = [
    WallRect(-2.425, 0.0,    5.0, 0.15,  math.pi / 2),
    WallRect( 2.425, 0.0,    5.0, 0.15, -math.pi / 2),
    WallRect( 0.0,   2.425,  5.0, 0.15,  0.0),
    WallRect( 0.0,  -2.425,  5.0, 0.15,  math.pi),
]

# Outer walls for the 7x7 stage 5 arena, walls at ±3.425.
OUTER_WALLS_7x7 = [
    WallRect(-3.425, 0.0,    7.0, 0.15,  math.pi / 2),
    WallRect( 3.425, 0.0,    7.0, 0.15, -math.pi / 2),
    WallRect( 0.0,   3.425,  7.0, 0.15,  0.0),
    WallRect( 0.0,  -3.425,  7.0, 0.15,  math.pi),
]

# Default inner-wall set used by stages 4, 5, 7-10 (5x5 arena, 7 segments).
INNER_WALLS = [
    WallRect(-2.0,  -1.5,    1.0, 0.15,  0.0),
    WallRect(-0.5,  -2.0,    1.0, 0.15, -math.pi / 2),
    WallRect( 1.0,  -1.0,    1.0, 0.15,  math.pi / 2),
    WallRect( 1.2,   1.9,    1.0, 0.15, -math.pi / 2),
    WallRect( 1.9,   0.4,    1.0, 0.15,  0.0),
    WallRect(-0.5,   1.5,    1.0, 0.15,  0.0),
    WallRect(-1.2,   0.092,  1.0, 0.15, -math.pi / 2),
]

# Stage 5 — 8 interior walls forming a maze inside the 7x7 arena.
# Matches tb3_stage5.sdf.
INNER_WALLS_STAGE5 = [
    WallRect(-1.5,   2.0,   1.0, 0.15,  0.0),
    WallRect( 1.5,   2.0,   1.0, 0.15,  0.0),
    WallRect(-2.0,   0.5,   1.0, 0.15,  math.pi / 2),
    WallRect( 0.5,   0.5,   1.0, 0.15,  math.pi / 2),
    WallRect( 0.0,  -1.5,   1.0, 0.15,  math.pi / 2),
    WallRect( 2.0,  -0.5,   1.0, 0.15,  math.pi / 2),
    WallRect(-1.5,  -2.5,   1.0, 0.15,  0.0),
    WallRect( 1.5,  -2.5,   1.0, 0.15,  0.0),
]

STATIC_CYLINDERS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
CYLINDER_RADIUS = 0.15

# Per-stage flags
HAS_INNER      = {1: False, 2: False, 3: False, 4: True,  5: True}
HAS_STATIC_CYL = {1: False, 2: True,  3: False, 4: False, 5: False}

# Per-stage half-extent of the rasterized map (= arena-side/2 + 1 m margin).
# Defaults to 2.8 m for the 5x5 arenas; stage 5 uses 3.8 m for its 7x7 arena.
HALF_EXTENT_DEFAULT = 2.8
HALF_EXTENT_BY_STAGE = {5: 3.8}

# Per-stage outer-wall set.
OUTER_WALLS_BY_STAGE = {5: OUTER_WALLS_7x7}

# Per-stage inner-wall overrides. Default is INNER_WALLS.
INNER_WALLS_BY_STAGE = {5: INNER_WALLS_STAGE5}


# ── rasterization ─────────────────────────────────────────────────────────────
def world_to_px(x: float, y: float, half_ext: float, size_px: int) -> tuple[int, int]:
    """Convert world coords to pixel coords for a map with the given extents."""
    px = int((x + half_ext) / RESOLUTION)
    py = (size_px - 1) - int((y + half_ext) / RESOLUTION)
    return px, py


def draw_rect(grid: np.ndarray, rect: WallRect,
              half_ext: float, size_px: int) -> None:
    """Rasterize a rotated rectangle into the grid as occupied."""
    cos_y = math.cos(rect.yaw)
    sin_y = math.sin(rect.yaw)
    hx = rect.size_x / 2 + INFLATE
    hy = rect.size_y / 2 + INFLATE

    max_r = math.hypot(hx, hy)
    x0, x1 = rect.cx - max_r, rect.cx + max_r
    y0, y1 = rect.cy - max_r, rect.cy + max_r
    px0, py0 = world_to_px(x0, y1, half_ext, size_px)
    px1, py1 = world_to_px(x1, y0, half_ext, size_px)
    px_lo, px_hi = min(px0, px1), max(px0, px1)
    py_lo, py_hi = min(py0, py1), max(py0, py1)

    for py in range(max(0, py_lo), min(size_px, py_hi + 1)):
        for px in range(max(0, px_lo), min(size_px, px_hi + 1)):
            wx = (px + 0.5) * RESOLUTION - half_ext
            wy = half_ext - (py + 0.5) * RESOLUTION
            dx, dy = wx - rect.cx, wy - rect.cy
            lx =  cos_y * dx + sin_y * dy
            ly = -sin_y * dx + cos_y * dy
            if abs(lx) <= hx and abs(ly) <= hy:
                grid[py, px] = OCCUPIED_VAL


def draw_circle(grid: np.ndarray, cx: float, cy: float, r: float,
                half_ext: float, size_px: int) -> None:
    rr = r + INFLATE
    px_lo, py_hi = world_to_px(cx - rr, cy - rr, half_ext, size_px)
    px_hi, py_lo = world_to_px(cx + rr, cy + rr, half_ext, size_px)
    for py in range(max(0, py_lo), min(size_px, py_hi + 1)):
        for px in range(max(0, px_lo), min(size_px, px_hi + 1)):
            wx = (px + 0.5) * RESOLUTION - half_ext
            wy = half_ext - (py + 0.5) * RESOLUTION
            if math.hypot(wx - cx, wy - cy) <= rr:
                grid[py, px] = OCCUPIED_VAL


def build_stage_map(stage: int) -> tuple[np.ndarray, float]:
    half_ext = HALF_EXTENT_BY_STAGE.get(stage, HALF_EXTENT_DEFAULT)
    size_px = int(2 * half_ext / RESOLUTION)
    grid = np.full((size_px, size_px), FREE_VAL, dtype=np.uint8)
    for w in OUTER_WALLS_BY_STAGE.get(stage, OUTER_WALLS_5x5):
        draw_rect(grid, w, half_ext, size_px)
    if HAS_INNER[stage]:
        for w in INNER_WALLS_BY_STAGE.get(stage, INNER_WALLS):
            draw_rect(grid, w, half_ext, size_px)
    if HAS_STATIC_CYL[stage]:
        for (cx, cy) in STATIC_CYLINDERS:
            draw_circle(grid, cx, cy, CYLINDER_RADIUS, half_ext, size_px)
    return grid, half_ext


# ── PGM + YAML output ─────────────────────────────────────────────────────────
def write_pgm(path: Path, grid: np.ndarray) -> None:
    h, w = grid.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(grid.tobytes())


def write_yaml(path: Path, pgm_name: str, half_ext: float) -> None:
    """Nav2 map YAML — origin is the world coord of the bottom-left pixel."""
    yaml_text = (
        f"image: {pgm_name}\n"
        f"resolution: {RESOLUTION}\n"
        f"origin: [{-half_ext}, {-half_ext}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    path.write_text(yaml_text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stage in range(1, 6):
        grid, half_ext = build_stage_map(stage)
        pgm_name = f"tb3_stage{stage}.pgm"
        write_pgm(OUT_DIR / pgm_name, grid)
        write_yaml(OUT_DIR / f"tb3_stage{stage}.yaml", pgm_name, half_ext)
        n_occ = int((grid == OCCUPIED_VAL).sum())
        print(f"  stage {stage:2d}: {grid.shape[1]}x{grid.shape[0]} px, "
              f"{n_occ:5d} occupied  ({pgm_name})")
    print(f"Done. Maps in {OUT_DIR}")


if __name__ == "__main__":
    main()
