#!/usr/bin/env python3
"""AIRL phase-2 arm of the interaction-budget sweep.

Trains PPO against the frozen recovered reward on a sim fleet, checkpointing
on the same budget grid as the TD3 arm so the two curves are directly
comparable.

A unit note that matters for reading the results. TD3 runs with
action_repeat=4, AIRL with 1, so at an equal number of BRIDGE steps the AIRL
policy has made four times as many decisions. That is not a bug in the
comparison: the budget axis is what a physical robot pays in interactions, and
action repeat is part of each method's design and its sample efficiency. The
comparison is at equal robot cost, not at equal decision count.

Prerequisites (regenerate with scripts/prepare_airl.sh -- the original
artifacts were never committed, since .gitignore excludes *.zip):

    reward_net.pt         recovered reward from train_airl.py
    policy_after_bc.zip   BC warm-start PPO checkpoint

Usage:

    scripts/launch_sim_fleet.sh -n 4 -s 4 --fast
    python3 /airl/sweep_airl.py --seed 0 --n-envs 4 \
        --reward-net  /checkpoints/airl_stage4/reward_net.pt \
        --init-policy /checkpoints/airl_stage4/policy_after_bc.zip \
        --out-dir     /checkpoints/sweep/airl_seed0
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
from stable_baselines3 import PPO

sys.path.insert(0, "/")
sys.path.insert(0, "/rl")

from airl.vec_airl_env import make_airl_fleet_vec_env  # noqa: E402
# Reuse the TD3 arm's callback rather than copying it, so both arms record
# budget points identically. Importing is side-effect-free (its main() is
# guarded) and, importantly, does not modify a file the running TD3 sweep
# re-executes for each remaining seed.
from sweep_td3 import DEFAULT_GRID, BudgetGridCheckpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/configs/airl.yaml")
    p.add_argument("--env-config", default="/configs/td3.yaml",
                   help="config carrying the env/ros2 sections. Defaults to "
                        "the TD3 config so both arms train in an identically "
                        "configured environment.")
    p.add_argument("--reward-net", required=True)
    p.add_argument("--init-policy", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=40)
    p.add_argument("--frame-stack", type=int, default=1,
                   help="must match what train_airl.py used for this reward net")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-bridge-steps", type=int, default=4_000_000)
    p.add_argument("--grid", type=int, nargs="*", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        airl_cfg = yaml.safe_load(fh)
    with open(args.env_config) as fh:
        env_cfg = yaml.safe_load(fh)

    ppo_cfg = airl_cfg.get("ppo", airl_cfg.get("phase2", {}))
    # AIRL phase 2 applies no action repeat, so one agent step is one bridge
    # step. Read it from config anyway so a future change cannot silently
    # desynchronise the budget accounting from reality.
    action_repeat = int(env_cfg.get("td3", {}).get("action_repeat", 1)) \
        if airl_cfg.get("use_action_repeat") else 1

    grid = [b for b in (args.grid or DEFAULT_GRID) if b <= args.max_bridge_steps]
    total_agent_steps = args.max_bridge_steps // action_repeat

    # PPO's update cadence is n_steps * n_envs env steps. Scaling n_steps down
    # by n_envs keeps one update per the same number of env steps as the
    # single-env reference; otherwise the vectorised run would update 4x less
    # often per interaction -- the on-policy analogue of the TD3 arm's
    # update-to-data problem.
    base_n_steps = int(ppo_cfg.get("n_steps", 1024))
    n_steps = max(1, base_n_steps // args.n_envs)
    rollout = n_steps * args.n_envs
    batch_size = int(ppo_cfg.get("batch_size", 64))
    if rollout % batch_size != 0:
        print(f"[sweep-airl] WARNING rollout {rollout} not divisible by "
              f"batch_size {batch_size}; SB3 will drop a partial minibatch.")

    print(f"[sweep-airl] seed={args.seed} n_envs={args.n_envs} "
          f"frame_stack={args.frame_stack} action_repeat={action_repeat}")
    print(f"[sweep-airl] n_steps {base_n_steps} -> {n_steps} per env "
          f"(rollout {rollout} env steps, unchanged update cadence)")
    print(f"[sweep-airl] grid (bridge steps): "
          f"{', '.join(f'{b:,}' for b in grid)}")

    if not rclpy.ok():
        rclpy.init()

    venv = make_airl_fleet_vec_env(
        env_cfg, args.n_envs,
        reward_net_path=args.reward_net,
        frame_stack=args.frame_stack,
        domain_base=args.domain_base,
    )

    # custom_objects overrides values baked into the warm-start checkpoint;
    # without it PPO.load would restore the single-env n_steps.
    model = PPO.load(
        args.init_policy,
        env=venv,
        custom_objects={
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": float(ppo_cfg.get("learning_rate", 3e-5)),
            "tensorboard_log": "/logs/tensorboard/airl_sweep",
        },
        seed=args.seed,
        verbose=1,
    )

    cb = BudgetGridCheckpoint(grid, action_repeat, out_dir)
    meta = {
        "arm": "airl_phase2",
        "seed": args.seed,
        "n_envs": args.n_envs,
        "action_repeat": action_repeat,
        "frame_stack": args.frame_stack,
        "n_steps_per_env": n_steps,
        "rollout_env_steps": rollout,
        "grid_bridge_steps": grid,
        "reward_net": args.reward_net,
        "init_policy": args.init_policy,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=total_agent_steps,
            callback=cb,
            reset_num_timesteps=True,
            tb_log_name=f"sweep_airl_seed{args.seed}",
        )
    except KeyboardInterrupt:
        print("\n[sweep-airl] interrupted -- reached checkpoints are intact.")
    finally:
        wall = time.perf_counter() - t0
        model.save(str(out_dir / "final"))
        (out_dir / "run_summary.json").write_text(json.dumps({
            **meta,
            "wall_total_h": round(wall / 3600, 3),
            "budgets_reached": sorted(cb.hit.keys()),
        }, indent=2))
        print(f"[sweep-airl] done in {wall / 3600:.2f} h; "
              f"{len(cb.hit)}/{len(grid)} budgets -> {out_dir}")
        venv.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
