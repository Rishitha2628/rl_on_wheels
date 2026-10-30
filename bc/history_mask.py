#!/usr/bin/env python3
"""Freeze the history of selected observation channels inside a frame stack.

Why this exists
---------------
Four stacked frames help the imitation policy by 9.3 points, and we do not know
which part of the history carries that. The 40-dim BC observation splits into

    [0..35]  lidar bins               \\  exteroception: where the obstacles are
    [39]     min range in a +-20 deg
             cone toward the goal     /

    [36]     normalised goal distance \\  self-motion: where the robot is
    [37]     cos(goal bearing, body)  |   relative to its goal. Across frames,
    [38]     sin(goal bearing, body)  /   the change in these IS the robot's own
                                         velocity, which is the only way this
                                         policy can know what it has been doing.

Note what is NOT in the list: the previous action. build_dataset.py omits
prev_lin_vel and prev_av deliberately (the policy would echo them), and
bc/eval.py slices the live 42-dim observation to its first 40 dims for the same
reason. So an "action history" ablation has nothing to ablate -- the stacked
policy has never seen an action. The self-motion group is the nearest thing the
observation actually contains.

To hold a group's history out without changing the network, we keep the input at
k*40 dims and overwrite that group's slots in every frame with the value from
the current frame. The policy still has the parameters to use history; it just
has no history to use for those channels. Capacity is therefore identical across
cells, so a difference is about information and not about model size.

The current frame is the LAST block of the stack, matching stack_frames() in
bc/train.py and the ring buffer in bc/eval.py.
"""
from __future__ import annotations

import numpy as np

# Channel groups within one 40-dim frame.
SCAN = tuple(list(range(36)) + [39])   # lidar bins + the goal-cone range
SELF = (36, 37, 38)                    # distance and bearing to the goal

GROUPS = {"scan": SCAN, "self": SELF}
BASE_DIM = 40


def resolve(keep: str) -> tuple[str, ...]:
    """Parse a --history value into the groups that keep their history."""
    keep = (keep or "all").strip().lower()
    if keep in ("all", ""):
        return tuple(GROUPS)
    if keep == "none":
        return ()
    out = []
    for name in keep.split("+"):
        name = name.strip()
        if name not in GROUPS:
            raise ValueError(f"unknown history group {name!r}; "
                             f"choose from {sorted(GROUPS)}, 'all' or 'none'")
        out.append(name)
    return tuple(out)


def frozen_indices(keep: str, base_dim: int = BASE_DIM) -> np.ndarray:
    """Columns within one frame whose history is frozen to the current frame."""
    kept = resolve(keep)
    live: set[int] = set()
    for name in kept:
        live.update(GROUPS[name])
    return np.array(sorted(set(range(base_dim)) - live), dtype=np.intp)


def apply(stacked: np.ndarray, keep: str, frame_stack: int,
          base_dim: int = BASE_DIM) -> np.ndarray:
    """Freeze the non-kept channels across a (..., frame_stack*base_dim) stack.

    Works on a single vector or a batch. Returns a new array; the input is not
    modified. A frame_stack of 1 is a no-op, since there is no history to hold.
    """
    if frame_stack <= 1:
        return stacked
    cols = frozen_indices(keep, base_dim)
    if cols.size == 0:
        return stacked
    out = np.array(stacked, dtype=np.float32, copy=True)
    flat = out.reshape(-1, frame_stack * base_dim)
    cur = (frame_stack - 1) * base_dim        # offset of the current frame
    for f in range(frame_stack - 1):          # every frame except the current
        flat[:, f * base_dim + cols] = flat[:, cur + cols]
    return out.reshape(stacked.shape)


def describe(keep: str, frame_stack: int, base_dim: int = BASE_DIM) -> str:
    kept = resolve(keep)
    frozen = frozen_indices(keep, base_dim)
    return (f"history: keep={keep or 'all'} -> groups {kept or '()'}; "
            f"{frozen.size} of {base_dim} channels frozen to the current frame "
            f"across {frame_stack} frames")
