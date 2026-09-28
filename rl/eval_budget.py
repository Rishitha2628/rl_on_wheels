#!/usr/bin/env python3
"""Evaluate every budget checkpoint of a sweep run and report success with CIs.

Produces one point per (method, seed, budget) on the crossover curve. Success
rate is a binomial proportion, so an interval is computed and reported
everywhere rather than quoting bare percentages that invite over-reading.

Episode count is not arbitrary. The BC baseline was first measured at 50
episodes: 62.0% with a 95% Wilson interval of [48.2%, 74.1%] -- 26 points
wide, so wide that no TD3 result could have cleared its upper bound and the
comparison would have been inconclusive by construction. Re-measured at 300
episodes it is 63.7% [58.1%, 68.9%], 10.8 points wide. Against that upper
bound, a CI-separated crossing needs 77% from 150 pooled TD3 episodes but only
74% from 300, which is why the default here is 100 per seed rather than 50.

Wilson rather than normal-approximation: the normal interval misbehaves badly
near 0 and 1, which is exactly where the early-budget points sit.

Assumes a fleet is up (evaluation is parallelised across it):

    scripts/launch_sim_fleet.sh -n 4 -s 4 --fast
    python3 /rl/eval_budget.py --run-dir /checkpoints/sweep/td3_seed0 \
        --episodes 100 --n-envs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from envs.vec_ros2_env import make_fleet_vec_env  # noqa: E402


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate(model, vec, n_episodes: int) -> dict:
    """Run n_episodes deterministic episodes across the vec env.

    Counts only the first n_episodes completions. A VecEnv auto-resets, so
    without a cap the faster instances would contribute disproportionately
    many episodes and bias the mean toward whichever env finishes quickest.
    """
    n_envs = vec.num_envs
    obs = vec.reset()

    successes = 0
    rewards: list[float] = []
    lengths: list[int] = []
    running_r = np.zeros(n_envs)
    running_l = np.zeros(n_envs, dtype=int)
    done_count = 0

    while done_count < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = vec.step(action)
        running_r += r
        running_l += 1

        for i, d in enumerate(dones):
            if not d or done_count >= n_episodes:
                continue
            info = infos[i]
            # VecMonitor stashes the true terminal stats under "episode";
            # info["is_success"] comes from our env's step().
            successes += int(bool(info.get("is_success")))
            ep = info.get("episode")
            rewards.append(float(ep["r"]) if ep else float(running_r[i]))
            lengths.append(int(ep["l"]) if ep else int(running_l[i]))
            running_r[i] = 0.0
            running_l[i] = 0
            done_count += 1

    lo, hi = wilson_interval(successes, done_count)
    return {
        "episodes": done_count,
        "successes": successes,
        "success_rate": round(successes / done_count, 4),
        "success_ci95": [round(lo, 4), round(hi, 4)],
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_length": round(float(np.mean(lengths)), 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="sweep output dir containing budget_index.json")
    p.add_argument("--config", default="/configs/td3.yaml")
    p.add_argument("--episodes", type=int, default=100,
                   help="episodes per checkpoint (default 100). Pooled across "
                        "3 seeds this gives 300 per budget point. Sized "
                        "against the BC baseline: BC measured 63.7%% over 300 "
                        "episodes, 95%% CI upper bound 68.9%%, so a "
                        "CI-separated crossing needs 74%% at 300 pooled "
                        "episodes versus 77%% at 150. The extra episodes cost "
                        "~30 min across the whole grid and lower the bar by "
                        "3 points.")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=40)
    p.add_argument("--out", default=None,
                   help="default: <run-dir>/eval_results.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    index_path = run_dir / "budget_index.json"

    if not index_path.exists():
        print(f"ERROR no budget_index.json in {run_dir}", file=sys.stderr)
        return 1

    budgets = json.loads(index_path.read_text())
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if not rclpy.ok():
        rclpy.init()

    # The arms differ in both algorithm and observation space, so the loader
    # and the env are chosen from the run's own metadata rather than assumed.
    # AIRL policies are PPO over the 40-dim Strip42to40 observation; TD3
    # policies are TD3 over the raw 42-dim one. Loading one with the other's
    # env fails on a space mismatch.
    arm = meta.get("arm", "td3")
    if arm.startswith("airl"):
        import sys as _sys
        _sys.path.insert(0, "/")
        from stable_baselines3 import PPO as Algo
        from airl.vec_airl_env import make_airl_eval_vec_env
        frame_stack = int(meta.get("frame_stack", 1))
        print(f"[eval] arm={arm}: PPO over the AIRL observation space "
              f"(frame_stack={frame_stack}), environment reward")
        vec = make_airl_eval_vec_env(config, args.n_envs,
                                     frame_stack=frame_stack,
                                     domain_base=args.domain_base)
    else:
        from stable_baselines3 import TD3 as Algo
        print(f"[eval] arm={arm}: TD3 over the raw observation space")
        vec = make_fleet_vec_env(config, args.n_envs,
                                 domain_base=args.domain_base)

    results = []

    try:
        for entry in budgets:
            ckpt = entry["checkpoint"]
            if not os.path.exists(ckpt):
                print(f"[eval] missing {ckpt}, skipping", file=sys.stderr)
                continue

            t0 = time.perf_counter()
            model = Algo.load(ckpt, env=vec)
            stats = evaluate(model, vec, args.episodes)
            stats.update({
                "budget_bridge_steps": entry["budget_bridge_steps"],
                "seed": meta.get("seed"),
                "checkpoint": ckpt,
                "eval_wall_s": round(time.perf_counter() - t0, 1),
            })
            results.append(stats)

            lo, hi = stats["success_ci95"]
            print(f"[eval] {entry['budget_bridge_steps']:>9,} steps: "
                  f"success {stats['success_rate']:.2%} "
                  f"[{lo:.2%}, {hi:.2%}]  "
                  f"reward {stats['mean_reward']:>8.1f}  "
                  f"len {stats['mean_length']:.0f}", flush=True)
    finally:
        vec.close()
        if rclpy.ok():
            rclpy.shutdown()

    out = Path(args.out) if args.out else run_dir / "eval_results.json"
    out.write_text(json.dumps(
        {"meta": meta, "episodes_per_point": args.episodes, "results": results},
        indent=2))
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
