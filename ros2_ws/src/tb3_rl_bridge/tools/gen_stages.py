#!/usr/bin/env python3
"""Generate stage SDF worlds for Ignition Fortress.

Reads the canonical scene/robot blocks from tb3_empty.sdf and emits
tb3_stage{1..10}.sdf next to it, each containing the same physics +
robot but a stage-specific obstacle layout.

Geometry:
  - outer arena: 5x5 m, walls at x/y = ±2.425
  - inner_walls: 7 segments of 1 m × 0.15 m × 0.5 m
  - static obstacles: 4 cylinders r=0.15, h=0.5 at (±1, ±1)
  - moving obstacles: cylinders r=0.16, h=0.5 — driven kinematically
                       by dynamic_obstacle_node (keyframe paths there)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

WORLDS_DIR = Path(__file__).resolve().parent.parent / "worlds"
TEMPLATE   = WORLDS_DIR / "tb3_empty.sdf"

# ── stage configuration ───────────────────────────────────────────────────────
# (inner_walls, static_obstacles, list_of_moving_obstacle_base_xy)
# Robot spawn pose per stage.
# Stages 1-3: arena center (0, 0). Stages 4-10: offset (-0.7, 0).
STAGE_SPAWN = {
    1: (0.0, 0.0, 0.0),
    2: (0.0, 0.0, 0.0),
    3: (0.0, 0.0, 0.0),
    4: (-0.7, 0.0, 0.0),
    5: (-0.7, 0.0, 0.0),
    6: (-0.7, 0.0, 0.0),
    7: (-0.7, 0.0, 0.0),
    8: (-0.7, 0.0, 0.0),
    9: (-0.7, 0.0, 0.0),
    10: (-0.7, 0.0, 0.0),
}

STAGES = {
    1:  (False, False, []),
    2:  (False, True,  []),
    # Stage 3: same 4 cylinders as stage 2 but MOVING with small-amplitude
    # oscillations (dynamic_obstacle_node + motion_mode="stage3").
    3:  (False, False, [(1, -1.0, -1.0), (2, -1.0, 1.0),
                        (3,  1.0, -1.0), (4,  1.0, 1.0)]),
    4:  (True,  False, [(1, 2.0, 2.0), (2, -2.0, -2.0)]),
    5:  (True,  False, [(1, 2, 2), (2, -2, -2), (3, 2, -2),
                        (4, -2, 2), (5, -2, 0), (6, 2, 0)]),
    6:  (False, False, [(1, 2, 2), (2, -2, -2), (3, 2, -2),
                        (4, -2, 2), (5, -2, 0), (6, 2, 0)]),
    7:  (True,  False, [(1, 2, 2), (2, -2, -2)]),
    8:  (True,  False, [(1, 2, 2), (2, -2, -2)]),
    9:  (True,  False, [(1, 2, 2), (2, -2, -2)]),
    10: (True,  False, [(1, 2, 2), (2, -2, -2)]),
}

# ── outer walls (5x5 arena) ───────────────────────────────────────────────────
OUTER_WALLS = """\
    <!-- ── outer walls (5x5 arena, walls at ±2.425) ── -->
    <model name="outer_wall_1">
      <static>true</static>
      <pose>-2.425 0 0 0 0 1.5708</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry></collision>
        <visual name="vis"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="outer_wall_2">
      <static>true</static>
      <pose>0 2.425 0 0 0 0</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry></collision>
        <visual name="vis"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="outer_wall_3">
      <static>true</static>
      <pose>2.425 0 0 0 0 -1.5708</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry></collision>
        <visual name="vis"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="outer_wall_4">
      <static>true</static>
      <pose>0 -2.425 0 0 0 3.14159</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry></collision>
        <visual name="vis"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>5 0.15 0.5</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

# ── inner walls (7 segments) ──────────────────────────────────────────────────
INNER_WALL_POSES = [
    (-2.0, -1.5, 0.0),
    (-0.5, -2.0, -1.5708),
    ( 1.0, -1.0,  1.5708),
    ( 1.2,  1.9, -1.5708),
    ( 1.9,  0.4,  0.0),
    (-0.5,  1.5,  0.0),
    (-1.2,  0.092, -1.5708),
]

def inner_walls_block() -> str:
    out = ["    <!-- ── inner walls (7 segments, 1m × 0.15m × 0.5m) ── -->"]
    for i, (x, y, yaw) in enumerate(INNER_WALL_POSES, 1):
        out.append(f"""\
    <model name="inner_wall_{i}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 {yaw}</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>1 0.15 0.5</size></box></geometry></collision>
        <visual name="vis"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>1 0.15 0.5</size></box></geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient><diffuse>0.6 0.4 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>""")
    return "\n".join(out) + "\n"

# ── static obstacles (4 cylinders at ±1, ±1) ──────────────────────────────────
STATIC_OBSTACLE_POSES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

def static_obstacles_block() -> str:
    out = ["    <!-- ── static obstacles (4 cylinders) ── -->"]
    for i, (x, y) in enumerate(STATIC_OBSTACLE_POSES, 1):
        out.append(f"""\
    <model name="static_obs_{i}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 0</pose>
      <link name="link"><pose>0 0 0.25 0 0 0</pose>
        <collision name="col">
          <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry></collision>
        <visual name="vis">
          <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.9 0.9 0.9 1</ambient><diffuse>0.9 0.9 0.9 1</diffuse></material>
        </visual>
      </link>
    </model>""")
    return "\n".join(out) + "\n"

# ── moving obstacles (positioned at base; motion is applied by mover node) ────
def moving_obstacles_block(specs: list[tuple[int, float, float]]) -> str:
    if not specs:
        return ""
    out = ["    <!-- ── moving obstacles (animated by dynamic_obstacle_node) ── -->"]
    for (idx, x, y) in specs:
        out.append(f"""\
    <model name="moving_obs_{idx}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 0</pose>
      <link name="link"><pose>0 0 0.25 0 0 0</pose>
        <collision name="col">
          <geometry><cylinder><radius>0.16</radius><length>0.5</length></cylinder></geometry></collision>
        <visual name="vis">
          <geometry><cylinder><radius>0.16</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.1 0.4 0.7 1</ambient><diffuse>0.1 0.4 0.7 1</diffuse></material>
        </visual>
      </link>
    </model>""")
    return "\n".join(out) + "\n"


def build_stage(stage_num: int, has_inner: bool, has_static: bool,
                moving: list[tuple[int, float, float]],
                header: str, robot: str) -> str:
    parts = [header, OUTER_WALLS]
    if has_inner:
        parts.append(inner_walls_block())
    if has_static:
        parts.append(static_obstacles_block())
    parts.append(moving_obstacles_block(moving))
    # Override the template's robot <pose> with the stage's canonical
    # spawn pose. Find the first <pose>...</pose> after the
    # `<model name="waffle_pi">` tag and rewrite it.
    spawn_x, spawn_y, spawn_theta = STAGE_SPAWN[stage_num]
    new_pose = f"      <pose>{spawn_x} {spawn_y} 0 0 0 {spawn_theta}</pose>"
    robot_lines = robot.split("\n")
    inside_waffle = False
    for i, line in enumerate(robot_lines):
        if 'name="waffle_pi"' in line:
            inside_waffle = True
            continue
        if inside_waffle and "<pose>" in line:
            robot_lines[i] = new_pose
            break
    robot = "\n".join(robot_lines)
    parts.append(robot)
    parts.append("\n  </world>\n</sdf>\n")
    return "".join(parts)


def main() -> None:
    src = TEMPLATE.read_text()

    # Header: everything up to (and including) ground_plane model.
    # Stops just before "Static cylinder obstacles" or "Arena walls".
    end_of_header = src.index("    <!-- ── Static cylinder obstacles")
    header = src[:end_of_header]
    header = header.replace(
        "Ignition Fortress world — empty environment with TurtleBot3 Waffle Pi.",
        "Ignition Fortress world — stage (auto-generated).",
    )

    # Robot: from "<!-- ── TurtleBot3 Waffle Pi ──" through its closing tag.
    robot_start = src.index("    <!-- ── TurtleBot3 Waffle Pi")
    # Closing </model> for the robot — there's only one </model> after </plugin>
    # at the very end, so we slice to the world close tag and trim.
    world_close = src.rindex("</world>")
    robot = src[robot_start:world_close]

    for stage_num, (has_inner, has_static, moving) in STAGES.items():
        out = build_stage(stage_num, has_inner, has_static, moving, header, robot)
        path = WORLDS_DIR / f"tb3_stage{stage_num}.sdf"
        path.write_text(out)
        print(f"Wrote {path.relative_to(WORLDS_DIR.parent.parent.parent.parent)} "
              f"(inner={has_inner} static={has_static} moving={len(moving)})")


if __name__ == "__main__":
    main()
