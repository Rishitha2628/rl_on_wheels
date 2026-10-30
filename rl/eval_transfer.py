#!/usr/bin/env python3
"""Evaluate one TD3 checkpoint in whatever world the running fleet is serving.

    python3 /rl/eval_transfer.py --checkpoint /checkpoints/tune/td3_confirm_seed0/budget_640000.zip \
        --config /configs/td3_tune_a.yaml --episodes 300 --out /checkpoints/transfer/x.json

This exists because eval_budget.py walks a whole budget grid from
budget_index.json, and a transfer test needs one trained policy evaluated in a
world it never saw. The policy keeps its OWN config -- in particular its own
max_lidar_range -- so only the geometry changes, not the observation scaling.
Changing both at once would confound "did it generalise" with "did its inputs
get rescaled".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.eval_budget import evaluate                    # noqa: E402
from rl.envs.vec_ros2_env import make_fleet_vec_env    # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True,
                   help="the config the policy was TRAINED with, not the "
                        "world's: observation scaling must stay as learned")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--domain-base", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="", help="recorded in the output")
    args = p.parse_args()

    import yaml
    from stable_baselines3 import TD3

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vec = make_fleet_vec_env(cfg, args.n_envs, domain_base=args.domain_base)
    try:
        model = TD3.load(args.checkpoint, env=None, device="auto")
        res = evaluate(model, vec, args.episodes)
    finally:
        vec.close()

    res.update(checkpoint=args.checkpoint, config=args.config, note=args.note)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"  {args.note or out.name}: {res['successes']}/{res['episodes']} "
          f"= {100*res['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
