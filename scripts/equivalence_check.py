#!/usr/bin/env python3
"""Test whether the accelerated sim arm is the same task as the real-time arm.

The interaction-budget experiment compares policies trained under a real-time
bridge (wall-clock paced, ~10 steps/s, matching what a physical TurtleBot3
allows) against policies trained under an unlimited-RTF bridge. That
comparison only means anything if the two bridges implement the *same MDP* and
differ only in how much wall time they consume.

This script tests that claim, and it does so against a noise floor rather than
against zero. The bridge is asynchronous -- sensor callbacks, DDS delivery and
the physics loop are not lockstepped -- so replaying an identical action
sequence twice in the SAME arm already produces slightly different
trajectories. Reporting rt-vs-fast divergence without that baseline would be
meaningless, since any nonzero number could be read as damning.

Protocol:

    # 1. noise floor -- same arm, twice, identical seed
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true
    python3 /scripts/equivalence_check.py record --arm rt  --run a
    python3 /scripts/equivalence_check.py record --arm rt  --run b

    # 2. the accelerated arm, same seed  (relaunch with fast:=true)
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true fast:=true
    python3 /scripts/equivalence_check.py record --arm fast --run a

    # 3. verdict
    python3 /scripts/equivalence_check.py compare

The verdict rule: fast-vs-rt divergence must sit within the rt-vs-rt noise
floor. If it is materially larger, the arms are different tasks and the
budget experiment is invalid until that is fixed.

Open-loop by construction -- a fixed action sequence, not a policy. A policy
would close the loop on the very divergence being measured, and small early
differences would amplify into completely different trajectories, telling us
nothing about whether the underlying dynamics match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(os.environ.get("EQUIV_DIR", "/logs/equivalence"))


def _load_env(config_path: str):
    """Import the ROS-backed env lazily.

    Only `record` touches the simulator. Keeping this out of module scope
    means `compare` is pure numpy and runs anywhere -- on the host, without
    ROS sourced, without a container -- which is where you actually want to
    look at results.
    """
    import yaml

    sys.path.insert(0, "/")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))
    from envs.ros2_gym_env import TurtleBot3Env

    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    return TurtleBot3Env(config)

# Observation layout, mirroring env_bridge_node's assemble_observation().
# Reported separately because they fail differently: LIDAR divergence means
# the world or the sensor desynced, GOAL divergence means the pose feed or
# reset path did, and PREV divergence would just echo our own action input.
BLOCKS = {
    "lidar": slice(0, 36),
    "goal": slice(36, 40),   # dist_norm, cos, sin, goal_path_min
    "prev_action": slice(40, 42),
}


def make_actions(n: int, seed: int, env) -> np.ndarray:
    """Fixed open-loop action sequence.

    Deliberately smooth rather than i.i.d. uniform: independent samples every
    step average out to near-zero motion and the robot barely leaves its spawn,
    which would exercise almost none of the dynamics we need to compare. A
    slow random walk drives it across the arena and into obstacle interactions.
    """
    rng = np.random.default_rng(seed)
    lo, hi = env.action_space.low, env.action_space.high
    mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0

    out = np.zeros((n, len(lo)), dtype=np.float32)
    x = np.zeros(len(lo), dtype=np.float64)
    for t in range(n):
        x = 0.92 * x + 0.08 * rng.normal(size=len(lo)) * 3.0
        out[t] = np.clip(mid + half * np.tanh(x), lo, hi).astype(np.float32)
    return out


def cmd_record(args) -> int:
    env = _load_env(args.config)
    actions = make_actions(args.steps, args.seed, env)

    obs0, _ = env.reset(seed=args.seed)
    observations = [obs0]
    rewards, dones = [], []

    for t in range(args.steps):
        obs, r, term, trunc, _ = env.step(actions[t])
        observations.append(obs)
        rewards.append(float(r))
        dones.append(bool(term or trunc))
        if term or trunc:
            # Re-reset with the SAME seed so both arms re-enter from the same
            # state; otherwise the sequences decorrelate at the first episode
            # boundary for reasons unrelated to sim pacing.
            obs, _ = env.reset(seed=args.seed)
            observations[-1] = obs

    env.close()

    out = DEFAULT_DIR / f"{args.arm}_{args.run}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.array(observations, dtype=np.float32),
        actions=actions,
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        seed=args.seed,
        arm=args.arm,
    )
    print(f"[equiv] recorded {len(observations)} obs -> {out}")
    return 0


def divergence(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    stats = {}
    for name, sl in BLOCKS.items():
        d = np.linalg.norm(a[:, sl] - b[:, sl], axis=1)
        stats[name] = {
            "mean": float(d.mean()),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
        }
    # Does divergence grow over the rollout (chaotic decorrelation) or stay
    # flat (bounded jitter)? Growth is the worrying signature.
    d_all = np.linalg.norm(a - b, axis=1)
    third = max(1, n // 3)
    stats["_trend"] = {
        "first_third_mean": float(d_all[:third].mean()),
        "last_third_mean": float(d_all[-third:].mean()),
        "steps_compared": int(n),
    }
    return stats


def cmd_compare(args) -> int:
    def load(arm, run):
        p = DEFAULT_DIR / f"{arm}_{run}.npz"
        if not p.exists():
            print(f"ERROR missing {p} -- run the record step for it first",
                  file=sys.stderr)
            return None
        return np.load(p)["observations"]

    cand = args.candidate
    rt_a, rt_b, cand_a = load("rt", "a"), load("rt", "b"), load(cand, "a")
    if rt_a is None or rt_b is None or cand_a is None:
        return 1

    floor = divergence(rt_a, rt_b)      # same arm twice = irreducible jitter
    cross = divergence(rt_a, cand_a)    # the comparison under test

    report = {
        "candidate": cand,
        "noise_floor_rt_vs_rt": floor,
        f"measured_rt_vs_{cand}": cross,
    }

    lbl = f"rt-vs-{cand}"
    print(f"\n  block         rt-vs-rt (floor)   {lbl:<14} ratio")
    print("  " + "-" * 56)
    verdict_ok = True
    for name in BLOCKS:
        f, c = floor[name]["mean"], cross[name]["mean"]
        ratio = (c / f) if f > 1e-9 else float("inf") if c > 1e-9 else 1.0
        flag = "" if ratio <= args.tolerance else "  <-- EXCEEDS"
        if ratio > args.tolerance:
            verdict_ok = False
        print(f"  {name:<12}  {f:>10.5f}       {c:>10.5f}   {ratio:>6.2f}x{flag}")

    t = cross["_trend"]
    growth = (t["last_third_mean"] / t["first_third_mean"]
              if t["first_third_mean"] > 1e-9 else float("inf"))
    print(f"\n  divergence trend over rollout: "
          f"{t['first_third_mean']:.5f} -> {t['last_third_mean']:.5f} "
          f"({growth:.2f}x)")
    print(f"  steps compared: {t['steps_compared']}")

    report["verdict"] = {
        "tolerance_ratio": args.tolerance,
        "within_noise_floor": bool(verdict_ok),
        "divergence_growth": None if growth == float("inf") else round(growth, 3),
    }

    if verdict_ok:
        print(f"\n  VERDICT: '{cand}' is within the noise floor of the "
              "real-time arm.\n  The budget comparison is valid.")
    else:
        print(f"\n  VERDICT: '{cand}' diverges beyond the noise floor.\n"
              "  Do NOT run the budget sweep until this is resolved -- the two\n"
              "  arms are different tasks, not the same task at different speeds.")

    out = DEFAULT_DIR / f"verdict_{cand}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {out}")
    return 0 if verdict_ok else 2


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record one open-loop rollout")
    rec.add_argument("--config", default="/configs/td3.yaml")
    rec.add_argument("--arm", required=True,
                     help="name for the bridge configuration currently running "
                          "(e.g. rt, fast, fleet4). Free-form so new "
                          "configurations can be compared without editing this.")
    rec.add_argument("--run", default="a",
                     help="run id; record rt twice (a, b) for the noise floor")
    rec.add_argument("--steps", type=int, default=600)
    rec.add_argument("--seed", type=int, default=0,
                     help="must be identical across every recording")
    rec.set_defaults(func=cmd_record)

    cmp_ = sub.add_parser("compare", help="report divergence vs noise floor")
    cmp_.add_argument("--candidate", default="fast",
                      help="arm to test against the rt noise floor "
                           "(default: fast). Needs <candidate>_a.npz recorded.")
    cmp_.add_argument("--tolerance", type=float, default=2.0,
                      help="max acceptable ratio of cross-arm divergence to "
                           "the same-arm noise floor (default 2.0)")
    cmp_.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
