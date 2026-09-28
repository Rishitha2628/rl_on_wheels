#!/usr/bin/env python3
"""Measure aggregate throughput of a running sim fleet.

Reports **bridge steps/s** so the number is directly comparable with
bench_throughput.py's single-instance figures (9.93 real-time, 58.2
accelerated). One vec.step() is n_envs x action_repeat bridge steps, and
conflating those units is the easy way to report a 4x speedup as 16x.

Usage (inside the container, fleet already up):

    scripts/launch_sim_fleet.sh -n 4 -s 4 --fast
    python3 /scripts/bench_fleet.py --config /configs/td3.yaml -n 4 \
        --steps 150 --out /logs/throughput/fleet4.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/configs/td3.yaml")
    p.add_argument("-n", "--n-envs", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=40)
    p.add_argument("--steps", type=int, default=150,
                   help="vec steps to time (each is n_envs agent steps)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    action_repeat = int(config.get("td3", {}).get("action_repeat", 1))

    from envs.vec_ros2_env import make_fleet_vec_env

    print(f"[fleet-bench] connecting to {args.n_envs} instance(s) "
          f"on domains {args.domain_base}..{args.domain_base + args.n_envs - 1}")
    vec = make_fleet_vec_env(config, args.n_envs, domain_base=args.domain_base)

    rng = np.random.default_rng(args.seed)
    low, high = vec.action_space.low, vec.action_space.high

    def sample():
        return rng.uniform(low=low, high=high,
                           size=(args.n_envs, low.shape[0])).astype(np.float32)

    vec.reset()
    for _ in range(args.warmup):
        vec.step(sample())

    latencies: list[float] = []
    t_start = time.perf_counter()
    for _ in range(args.steps):
        t0 = time.perf_counter()
        vec.step(sample())
        latencies.append(time.perf_counter() - t0)
    wall_total = time.perf_counter() - t_start

    vec.close()

    agent_steps = args.steps * args.n_envs
    bridge_steps = agent_steps * action_repeat
    bridge_per_sec = bridge_steps / wall_total

    result = {
        "n_envs": args.n_envs,
        "action_repeat": action_repeat,
        "vec_steps": args.steps,
        "agent_steps": agent_steps,
        "bridge_steps": bridge_steps,
        "wall_total_s": round(wall_total, 3),
        "bridge_steps_per_sec": round(bridge_per_sec, 2),
        "agent_steps_per_sec": round(agent_steps / wall_total, 2),
        "per_instance_bridge_steps_per_sec": round(bridge_per_sec / args.n_envs, 2),
        "vec_step_latency_ms": {
            "mean": round(1000 * statistics.mean(latencies), 2),
            "p50": round(1000 * statistics.median(latencies), 2),
            "p95": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 2),
        },
        "hours_per_1M_bridge_steps": round(1e6 / bridge_per_sec / 3600, 2),
        # The configured TD3 budget, in wall-clock hours at this throughput.
        "hours_for_td3_4M_bridge_steps": round(4e6 / bridge_per_sec / 3600, 2),
    }

    print(json.dumps(result, indent=2))

    # Scaling efficiency against the measured single-instance accelerated
    # figure. Below ~0.8 means the instances are fighting for CPU and adding
    # more will not help.
    single = 58.21
    eff = result["per_instance_bridge_steps_per_sec"] / single
    print(f"\n[fleet-bench] {bridge_per_sec:.1f} bridge steps/s aggregate "
          f"({result['per_instance_bridge_steps_per_sec']:.1f}/instance)")
    print(f"[fleet-bench] scaling efficiency vs 1 instance ({single}/s): {eff:.2f}")
    print(f"[fleet-bench] TD3 4M-step run: "
          f"{result['hours_for_td3_4M_bridge_steps']:.1f} h")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[fleet-bench] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
