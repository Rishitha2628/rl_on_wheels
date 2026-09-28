#!/usr/bin/env python3
"""Train TD3 on a sim fleet, checkpointing on an interaction-budget grid.

One run of this produces one (method, seed) curve for the budget sweep: a
series of checkpoints taken at prescribed numbers of environment
interactions, so a single long run yields the whole budget axis rather than
needing a separate run per budget.

Budgets are counted in **bridge steps** -- actual robot-environment
interactions -- not in SB3 timesteps. With action_repeat=4 those differ by 4x,
and the paper's claim is denominated in robot-hours, so the unit has to be the
one a real robot would pay.

Assumes the fleet is already up:

    scripts/launch_sim_fleet.sh -n 4 -s 4 --fast
    python3 /rl/sweep_td3.py --config /configs/td3.yaml --seed 0 \
        --n-envs 4 --out-dir /checkpoints/sweep/td3_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import rclpy
import yaml
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(__file__))

from agents.td3_agent import build  # noqa: E402
from envs.vec_ros2_env import make_fleet_vec_env  # noqa: E402

# Interaction budgets in BRIDGE steps. Roughly log-spaced, starting at
# learning_starts (40k bridge = 10k agent steps) since everything below that
# is a random policy and carries no seed-dependent information.
DEFAULT_GRID = [
    40_000, 80_000, 160_000, 320_000, 640_000,
    1_280_000, 2_000_000, 3_000_000, 4_000_000,
]


class BudgetGridCheckpoint(BaseCallback):
    """Save a checkpoint the first time each budget threshold is crossed.

    SB3's CheckpointCallback only does fixed intervals. A log-spaced grid puts
    resolution where the learning curve actually moves (early) without paying
    for it at the flat end, and it doubles as crash insurance: an interrupted
    run still yields every budget point it already reached.
    """

    def __init__(self, grid_bridge: list[int], action_repeat: int,
                 out_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.grid_bridge = sorted(grid_bridge)
        self.action_repeat = action_repeat
        self.out_dir = out_dir
        self.hit: dict[int, dict] = {}
        self._t0 = time.perf_counter()

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        # num_timesteps counts agent steps summed across all envs.
        bridge = self.model.num_timesteps * self.action_repeat

        for b in self.grid_bridge:
            if b in self.hit or bridge < b:
                continue
            path = self.out_dir / f"budget_{b}"
            self.model.save(str(path))
            elapsed = time.perf_counter() - self._t0
            self.hit[b] = {
                "budget_bridge_steps": b,
                "actual_bridge_steps": int(bridge),
                "agent_steps": int(self.model.num_timesteps),
                "wall_s": round(elapsed, 1),
                "bridge_steps_per_sec": round(bridge / max(elapsed, 1e-9), 2),
                "checkpoint": str(path) + ".zip",
            }
            if self.verbose:
                print(f"[budget] {b:>9,} bridge steps reached at "
                      f"{elapsed / 3600:.2f} h -> {path.name}.zip", flush=True)
            (self.out_dir / "budget_index.json").write_text(
                json.dumps(list(self.hit.values()), indent=2))
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/configs/td3.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=40)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-bridge-steps", type=int, default=4_000_000,
                   help="stop here; must be >= the largest grid point wanted")
    p.add_argument("--grid", type=int, nargs="*", default=None,
                   help="budget grid in bridge steps (default: log-spaced)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    action_repeat = int(config["td3"].get("action_repeat", 1))
    grid = [b for b in (args.grid or DEFAULT_GRID) if b <= args.max_bridge_steps]

    # SB3 counts total_timesteps in agent steps across all envs.
    total_agent_steps = args.max_bridge_steps // action_repeat

    # Hold the update-to-data ratio fixed. SB3 collects n_envs transitions per
    # round; without this the vectorised run would do 1 gradient step per
    # n_envs transitions while the reference config does 1 per transition.
    base_grad_steps = int(config["td3"]["gradient_steps"])
    grad_steps = base_grad_steps * args.n_envs

    print(f"[sweep] seed={args.seed} n_envs={args.n_envs} "
          f"action_repeat={action_repeat}")
    print(f"[sweep] budget grid (bridge steps): "
          f"{', '.join(f'{b:,}' for b in grid)}")
    print(f"[sweep] total: {args.max_bridge_steps:,} bridge = "
          f"{total_agent_steps:,} agent steps")
    print(f"[sweep] gradient_steps {base_grad_steps} -> {grad_steps} "
          f"to preserve update-to-data ratio at n_envs={args.n_envs}")

    if not rclpy.ok():
        rclpy.init()

    env = make_fleet_vec_env(config, args.n_envs, domain_base=args.domain_base)
    env.seed(args.seed)

    model = build(config, env, seed=args.seed,
                  gradient_steps_override=grad_steps)

    cb = BudgetGridCheckpoint(grid, action_repeat, out_dir)

    meta = {
        "seed": args.seed,
        "n_envs": args.n_envs,
        "action_repeat": action_repeat,
        "gradient_steps": grad_steps,
        "grid_bridge_steps": grid,
        "max_bridge_steps": args.max_bridge_steps,
        "config": args.config,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=total_agent_steps,
            callback=cb,
            log_interval=config["training"]["log_interval"],
            tb_log_name=f"sweep_td3_seed{args.seed}",
        )
    except KeyboardInterrupt:
        print("\n[sweep] interrupted -- checkpoints already reached are intact.")
    finally:
        wall = time.perf_counter() - t0
        model.save(str(out_dir / "final"))
        summary = {
            **meta,
            "wall_total_h": round(wall / 3600, 3),
            "budgets_reached": sorted(cb.hit.keys()),
            "mean_bridge_steps_per_sec": round(
                model.num_timesteps * action_repeat / max(wall, 1e-9), 2),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[sweep] done in {wall / 3600:.2f} h; "
              f"{len(cb.hit)}/{len(grid)} budgets reached -> {out_dir}")
        env.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
