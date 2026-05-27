#!/usr/bin/env python3
"""Behavior cloning training — supervised MSE on (obs, action) pairs.

Loads the NPZ produced by build_dataset.py, trains an MLP to imitate the
Nav2 demonstrator. No env interaction during training (pure supervised).

Usage:
    python3 /bc/train.py --config /configs/bc.yaml --dataset /demos/stage4_bc.npz
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from .networks import BCPolicy


# ── action rescaling ──────────────────────────────────────────────────────────
def normalize_action(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """[low, high] → [-1, 1] — what the policy is trained to output."""
    return 2.0 * (a - low) / (high - low) - 1.0


# ── training ──────────────────────────────────────────────────────────────────
def train(cfg_path: str, dataset_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    bc_cfg     = cfg["bc"]
    train_cfg  = cfg["training"]
    env_cfg    = cfg["env"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── load dataset ──────────────────────────────────────────────────────────
    data = np.load(dataset_path)
    obs    = data["obs"].astype(np.float32)
    action = data["action"].astype(np.float32)
    print(f"loaded {len(obs)} transitions from {dataset_path}")

    # Rescale demonstrator actions from env bounds → [-1, 1] (tanh output range)
    act_low  = np.array([env_cfg["min_linear_vel"], -env_cfg["max_angular_vel"]],
                        dtype=np.float32)
    act_high = np.array([env_cfg["max_linear_vel"],  env_cfg["max_angular_vel"]],
                        dtype=np.float32)
    action_norm = normalize_action(action, act_low, act_high)
    action_norm = np.clip(action_norm, -1.0, 1.0)

    obs_t    = torch.from_numpy(obs)
    action_t = torch.from_numpy(action_norm)

    dataset = TensorDataset(obs_t, action_t)
    val_frac = float(bc_cfg.get("val_fraction", 0.1))
    n_val    = max(1, int(len(dataset) * val_frac))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    batch_size = int(bc_cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    obs_dim = obs.shape[1]
    act_dim = action.shape[1]
    model = BCPolicy(obs_dim, act_dim,
                     hidden=bc_cfg.get("net_arch", [512, 512])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(bc_cfg["learning_rate"]))
    loss_fn = nn.MSELoss()

    os.makedirs(train_cfg["checkpoint_dir"],  exist_ok=True)
    os.makedirs(train_cfg["tensorboard_log"], exist_ok=True)
    writer = SummaryWriter(log_dir=train_cfg["tensorboard_log"])

    n_epochs = int(bc_cfg["epochs"])
    log_every = int(train_cfg.get("log_interval", 100))
    save_every = int(train_cfg.get("save_every_epoch", 5))

    global_step = 0
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses: list[float] = []
        for batch_idx, (o, a) in enumerate(train_loader):
            o = o.to(device); a = a.to(device)
            pred = model(o)
            loss = loss_fn(pred, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
            writer.add_scalar("train/mse", loss.item(), global_step)
            global_step += 1
            if batch_idx % log_every == 0:
                print(f"  [epoch {epoch:3d} batch {batch_idx:5d}] "
                      f"train_mse={loss.item():.4f}")

        # ── val ───────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_losses: list[float] = []
            for o, a in val_loader:
                o = o.to(device); a = a.to(device)
                pred = model(o)
                val_losses.append(float(loss_fn(pred, a).item()))
        val_mse = float(np.mean(val_losses))
        train_mse = float(np.mean(train_losses))
        writer.add_scalar("val/mse", val_mse, epoch)
        elapsed = time.time() - t0
        print(f"[epoch {epoch:3d}] train_mse={train_mse:.4f} "
              f"val_mse={val_mse:.4f} elapsed={elapsed:.1f}s")

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_mse < best_val:
            best_val = val_mse
            best_path = Path(train_cfg["checkpoint_dir"]) / "bc_best.pt"
            torch.save({"model":    model.state_dict(),
                        "obs_dim":  obs_dim,
                        "act_dim":  act_dim,
                        "act_low":  act_low,
                        "act_high": act_high,
                        "val_mse":  val_mse}, best_path)
            print(f"  ★ new best — saved {best_path}")

        if epoch % save_every == 0:
            ckpt = Path(train_cfg["checkpoint_dir"]) / f"bc_epoch{epoch}.pt"
            torch.save({"model":    model.state_dict(),
                        "obs_dim":  obs_dim,
                        "act_dim":  act_dim,
                        "act_low":  act_low,
                        "act_high": act_high}, ckpt)

    print(f"done. best val_mse = {best_val:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="/configs/bc.yaml")
    p.add_argument("--dataset", required=True, help="path to .npz from build_dataset.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config, args.dataset)


if __name__ == "__main__":
    main()
