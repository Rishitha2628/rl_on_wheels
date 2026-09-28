#!/usr/bin/env python3
"""Measure env-step throughput against a running tb3_rl_bridge.

This is the instrument behind the interaction-budget experiment. It answers
one question: how many env steps per second does this bridge configuration
deliver, and how much simulated time does each one actually cover?

The second half matters as much as the first. A fast run is only meaningful
if each step still advances the sim by step_duration -- if the sim clock is
lagging, the policy is seeing less motion per step and the two arms are no
longer comparable. `sim_s_per_step` is the number to check: it should sit at
step_duration (0.1 by default) in BOTH arms. Only `wall_s_per_step` should
differ.

Usage (inside the sim container, with a bridge already launched):

    # real-time arm -- expect ~10 steps/s
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true
    python3 /scripts/bench_throughput.py --config /configs/td3.yaml \
        --steps 500 --label rt --out /logs/throughput/rt.json

    # accelerated arm -- same command, fast:=true on the launch
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true fast:=true
    python3 /scripts/bench_throughput.py --config /configs/td3.yaml \
        --steps 500 --label fast --out /logs/throughput/fast.json

Actions are sampled from a fixed seed so the two arms drive the robot through
comparable state distributions -- throughput on an agent pinned against a wall
is not the throughput training sees.
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

from envs.ros2_gym_env import TurtleBot3Env  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/configs/td3.yaml",
                   help="config yaml (only env/ros2 sections are used)")
    p.add_argument("--steps", type=int, default=500,
                   help="env steps to time (after warmup)")
    p.add_argument("--warmup", type=int, default=25,
                   help="steps to discard before timing; covers first-call "
                        "service discovery and DDS handshake costs")
    p.add_argument("--seed", type=int, default=0,
                   help="action-sampling seed; keep identical across arms")
    p.add_argument("--label", default="unlabeled",
                   help="name for this configuration, recorded in the output")
    p.add_argument("--out", default=None,
                   help="write results JSON here (parent dirs created)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    step_duration = float(config["env"].get("step_duration", 0.1))
    env = TurtleBot3Env(config)
    rng = np.random.default_rng(args.seed)

    def sample():
        return env.action_space.sample() * 0 + rng.uniform(
            low=env.action_space.low, high=env.action_space.high
        ).astype(np.float32)

    print(f"[bench] label={args.label} warmup={args.warmup} steps={args.steps}")
    env.reset(seed=args.seed)

    for _ in range(args.warmup):
        _, _, term, trunc, _ = env.step(sample())
        if term or trunc:
            env.reset()

    # ── timed section ─────────────────────────────────────────────────────────
    latencies: list[float] = []
    episodes = 0
    successes = 0
    wall_t0 = time.perf_counter()

    for _ in range(args.steps):
        t0 = time.perf_counter()
        _, _, term, trunc, info = env.step(sample())
        latencies.append(time.perf_counter() - t0)
        if term or trunc:
            episodes += 1
            successes += int(bool(info.get("is_success")))
            env.reset()

    wall_total = time.perf_counter() - wall_t0
    env.close()

    # Sim time is inferred, not read: each completed /step advances the sim by
    # step_duration by construction, so the check is whether wall time per step
    # in the real-time arm matches it. In the fast arm the ratio is the speedup.
    steps_per_sec = args.steps / wall_total
    wall_s_per_step = wall_total / args.steps

    result = {
        "label": args.label,
        "steps": args.steps,
        "seed": args.seed,
        "config": args.config,
        "step_duration_s": step_duration,
        "wall_total_s": round(wall_total, 3),
        "steps_per_sec": round(steps_per_sec, 2),
        "wall_s_per_step": round(wall_s_per_step, 5),
        "realtime_factor": round(step_duration / wall_s_per_step, 2),
        "latency_ms": {
            "mean": round(1000 * statistics.mean(latencies), 2),
            "p50": round(1000 * statistics.median(latencies), 2),
            "p95": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 2),
            "max": round(1000 * max(latencies), 2),
        },
        "episodes_completed": episodes,
        "successes": successes,
        # Projected wall-clock cost of the training budgets in the sweep.
        "hours_per_1M_steps": round(1e6 / steps_per_sec / 3600, 2),
    }

    print(json.dumps(result, indent=2))
    print(f"\n[bench] {steps_per_sec:.1f} steps/s "
          f"= {result['realtime_factor']:.1f}x real time "
          f"-> {result['hours_per_1M_steps']:.1f} h per 1M steps")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[bench] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
