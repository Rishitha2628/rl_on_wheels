#!/usr/bin/env python3
"""Compare the reward AIRL recovered against the one we hand-wrote.

    python3 /airl/compare_rewards.py --reward-net /checkpoints/airl_s5/reward_net.pt \
        --demos /demos/stage11_dagger_agg2.npz --out /checkpoints/reward_cmp

Two reward functions are not directly comparable: adding a constant, scaling by
a positive factor, or adding potential-based shaping all leave the optimal
policy unchanged, and AIRL's architecture f(s,a,s') = g(s,a) + gamma*h(s') - h(s)
contains such a shaping term by construction. So we compare two things that
survive those transformations:

  1. SHAPE. Sweep one observation dimension and plot both rewards, each
     z-scored over the sweep. Scale and offset drop out; the shape does not.
  2. ORDERING. Score states the robot actually visited with both rewards and
     take Spearman's rank correlation, which is invariant to any monotone
     rescaling. This is the question a reward is for: does it rank states the
     same way?

One term is deliberately excluded. The hand-written r_obstacle fires on the
geometric distance to the moving cylinders, read from the simulator's pose
topic -- privileged information that never enters the 40-dim observation. AIRL
cannot recover a term it cannot see, so comparing against it would be unfair;
we compare against the observation-computable part and say so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airl.inspect_reward import (baseline_obs, _idx, query_reward,   # noqa: E402
                                 sweep_distance, sweep_lidar, sweep_goal_angle)

MAX_LV = 0.22
W_VLINEAR, W_VANGULAR = 10.0, 1.0


def analytic_reward(obs: np.ndarray, act: np.ndarray, fs: int = 1) -> np.ndarray:
    """The hand-written reward, minus the term that needs privileged state.

    Mirrors env_bridge_node.cpp:
        r_yaw      = -|goal_body|
        r_vangular = -w_ang * av^2
        r_vlinear  = -((max_lv - lv) * w_lin)^2
        r_distance = 2*d0/(d0+d) - 1          -> monotone in d; we use the
                                                 normalised distance directly,
                                                 since d0 is per-episode and
                                                 only rescales it
        r_obstacle = -20 if geometric dist < 0.22   [EXCLUDED: not in obs]
    """
    i36, i37, i38 = _idx(36, fs), _idx(37, fs), _idx(38, fs)
    goal_body = np.arctan2(obs[:, i38], obs[:, i37])
    r_yaw = -np.abs(goal_body)
    r_vangular = -W_VANGULAR * act[:, 1] ** 2
    lv_diff = MAX_LV - act[:, 0]
    r_vlinear = -((lv_diff * W_VLINEAR) ** 2)
    r_distance = 2.0 / (1.0 + obs[:, i36]) - 1.0
    return r_yaw + r_vangular + r_vlinear + r_distance - 1.0


def batch_reward(fn, obs: np.ndarray, act: np.ndarray,
                 next_obs: np.ndarray) -> np.ndarray:
    """Vectorised query. `fn` is either the net (shaped: g + gamma*h' - h) or
    `net.base` (just g). Only g is what AIRL claims to recover -- the potential
    h is exactly the part that is not identified -- so g is the comparison we
    report, and we record both to show the choice does not matter here."""
    with torch.no_grad():
        return fn(torch.from_numpy(obs), torch.from_numpy(act),
                  torch.from_numpy(next_obs), torch.zeros(len(obs))).numpy()


def z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (its numpy pin is broken in this image)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                   help="AIRL run dir holding reward_net.pt and metadata.yaml")
    p.add_argument("--demos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-states", type=int, default=20000)
    args = p.parse_args()

    # Rebuild exactly as train_airl.py did, or the state_dict keys will not
    # line up (RunningNorm adds buffers the plain constructor lacks).
    import gymnasium as gym
    import yaml
    from pathlib import Path
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    from imitation.util.networks import RunningNorm

    ck = Path(args.checkpoint_dir)
    meta = yaml.safe_load(open(ck / "metadata.yaml"))
    obs_space = gym.spaces.Box(np.array(meta["obs_low"], dtype=np.float32),
                               np.array(meta["obs_high"], dtype=np.float32),
                               dtype=np.float32)
    act_space = gym.spaces.Box(np.array(meta["act_low"], dtype=np.float32),
                               np.array(meta["act_high"], dtype=np.float32),
                               dtype=np.float32)
    net = BasicShapedRewardNet(observation_space=obs_space,
                               action_space=act_space,
                               normalize_input_layer=RunningNorm)
    net.load_state_dict(torch.load(ck / "reward_net.pt", map_location="cpu"))
    net.eval()

    obs_dim = int(meta["obs_dim"])
    fs = max(1, obs_dim // 40)          # 40 base dims per stacked frame

    # ── 1. shape, z-scored so scale and offset drop out ────────────────────
    d = np.load(args.demos)
    O, A = d["obs"].astype(np.float32), d["action"].astype(np.float32)
    if fs > 1:                           # the net expects stacked input
        O = np.tile(O, (1, fs))
    dim = obs_dim

    sweeps = {}
    for name, fn, lbl in (("distance", sweep_distance, "normalised goal distance"),
                          ("lidar", sweep_lidar, "LiDAR clearance"),
                          ("goal_angle", sweep_goal_angle, "goal bearing (rad)")):
        xs, airl = fn(net, dim, fs)
        xs_b, airl_base = fn(net.base, dim, fs)   # g only
        synth = np.tile(baseline_obs(dim, fs), (len(xs), 1))
        i = {"distance": _idx(36, fs), "lidar": _idx(39, fs), "goal_angle": None}[name]
        if name == "distance":
            synth[:, i] = xs
        elif name == "lidar":
            off = 40 * (fs - 1)
            synth[:, off:off + 36] = xs[:, None]; synth[:, i] = xs
        else:
            synth[:, _idx(37, fs)] = np.cos(xs); synth[:, _idx(38, fs)] = np.sin(xs)
        hand = analytic_reward(synth, np.tile([0.15, 0.0], (len(xs), 1)).astype(np.float32), fs)
        sweeps[name] = dict(x=xs.tolist(), airl=z(airl).tolist(),
                            airl_base=z(airl_base).tolist(),
                            hand=z(hand).tolist(), xlabel=lbl,
                            hand_is_constant=bool(np.std(hand) < 1e-9))

    # ── 2. ordering, on states the robot actually visited ──────────────────
    n = min(args.n_states, len(O))
    sel = np.random.default_rng(0).choice(len(O), n, replace=False)
    Os, As = O[sel], A[sel]
    shaped_r = batch_reward(net,      Os, As, Os)
    base_r   = batch_reward(net.base, Os, As, Os)
    hand_r   = analytic_reward(Os, As, fs)
    rho        = spearman(base_r, hand_r)       # headline: g vs hand
    rho_shaped = spearman(shaped_r, hand_r)
    rho_sb     = spearman(shaped_r, base_r)
    print(f"  rho(base g, hand)   = {rho:+.3f}")
    print(f"  rho(shaped, hand)   = {rho_shaped:+.3f}")
    print(f"  rho(shaped, base)   = {rho_sb:+.3f}")

    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(sweeps=sweeps, spearman=rho, spearman_shaped=rho_shaped,
                   spearman_shaped_vs_base=rho_sb, n_states=n,
                   excluded_term="r_obstacle (needs geometric obstacle distance, "
                                 "not present in the observation)"),
              open(os.path.join(args.out, "reward_comparison.json"), "w"), indent=2)
    print(f"  Spearman rank correlation over {n:,} visited states: rho = {rho:+.3f}")
    print(f"  wrote {args.out}/reward_comparison.json")


if __name__ == "__main__":
    main()
