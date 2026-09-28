# Results

All numbers below are stage 4 (5x5 m arena, 7 inner walls, 2 moving cylinders)
unless noted. Evaluation is a deterministic rollout; success means reaching the
goal within 0.40 m, which is checked in the C++ bridge and is the same for every
method. Intervals are Wilson 95% intervals on the binomial proportion.

## Main table

| Method | Success | 95% CI | Episodes |
| ------ | ------- | ------ | -------- |
| BC + safety overrides | 78.0 % | [73.0, 82.3] | 300 |
| TD3, softened reward shaping | 69.0 % | [59, 77] | 100 |
| TD3, lr 3e-4 / sigma 0.1 (3 seeds) | 68.0 % | [63, 73] | 300 |
| BC policy alone | 63.7 % | [58.1, 68.9] | 300 |
| TD3, lr 1e-4 / sigma 0.05 (3 seeds) | 62.7 % | [57, 68] | 300 |
| AIRL phase 2, best point | 23.7 % | [19, 29] | 300 |

The BC policy is the same network in both BC rows. The difference is the
eval-time wrapper in `bc/eval.py`: two clearance overrides and a P-controller
for the last 0.3 m. Those are worth 14.3 points. The intervals do not overlap.

## Success against interaction budget

The bridge sleeps for the control period on every step, so it runs at 10 Hz
regardless of how fast the machine is. Measured throughput is 9.93 steps/s,
which is what the robot-hours column converts at.

`action_repeat` is 4 in `configs/td3.yaml`, so a configured 1M timesteps is 4M
environment steps. The budgets below are environment steps.

| Budget | Robot-hours | TD3 (lr 1e-4) | TD3 (lr 3e-4) | AIRL |
| ------ | ----------- | ------------- | ------------- | ---- |
| 40 k | 1.1 | 0.0 % | 0.0 % | 23.7 % |
| 80 k | 2.2 | 32.3 % | 37.0 % | 16.3 % |
| 160 k | 4.5 | 54.7 % | 60.0 % | 3.3 % |
| 320 k | 9.0 | 62.0 % | 64.0 % | 2.0 % |
| 640 k | 17.9 | 62.7 % | 68.0 % | 3.0 % |
| 1.28 M | 35.8 | 64.5 % | -- | 1.0 % |
| 2 M | 55.9 | 59.5 % | -- | 0.5 % |

Neither TD3 configuration clears the upper bound of BC's interval (68.9 %) at
any budget, so there is no interval-separated crossing. The tuned runs stop at
640 k because they were capped there.

Figure: `paper/figures/crossover.pdf` (not tracked; regenerate with
`scripts/analyze_crossover.py`).

## Hyperparameters

`configs/td3.yaml` ships learning rate 1e-4 and OU sigma 0.05. Its comments say
both were lowered to fine-tune a policy that had already converged and had
plateaued near 0.59. Training from scratch with them reproduces roughly that
plateau. Running the SB3 defaults instead (3e-4, 0.1) over three seeds gives
68.0 %, about five points better.

One seed of that configuration read 79 % at 640 k. It did not replicate: the
three confirmation seeds gave 65 / 68 / 71 at the same budget. Seed variance is
much larger at small budgets, e.g. 4 / 61 / 46 % at 80 k.

`configs/td3_tune_a.yaml` and `td3_tune_b.yaml` hold the two variants.

## Reward shaping

Per environment step the shaping terms are roughly

```
r_yaw       -3.14 .. 0
r_vangular  -4.00 .. 0     -w * omega^2, omega up to 2.0
r_vlinear   -4.84 .. 0     -((0.22 - v) * scale)^2, worst at v = 0
constant    -1
```

against a terminal penalty of -2000 for a collision. A robot that slows and
turns therefore accrues about -13 per step, or about -5850 over a 450-step
episode, so slowing down costs more than crashing. That looked like a plausible
cause of the plateau.

It is not. `r_vlinear_scale` and `r_vangular_weight` are launch parameters
(defaults 10.0 and 1.0 reproduce the original reward). Running 5.0/0.5 and
2.5/0.25 gives 69 % and 69 % against the unmodified 68 %.

## AIRL

Phase 2 peaks at 23.7 % on the smallest budget and falls to ~0.5 %.

The first run of this was worse (about 1 % throughout) because of where the
reward net was saved. `train_airl.py` wrote it at the end of the adversarial
loop, and over that loop the generator went 0.50 -> 0.30 -> 0.20 -> 0.10. Phase
2 was then optimising a reward fitted against a collapsed policy. The script now
snapshots the reward net alongside the best-eval policy and `prepare_airl.sh`
promotes that snapshot; the generator peaked at 0.80 on the rerun and the early
budget points improved. Phase 2 still walks away from the warm start.

## Earlier stage-5 numbers

From earlier runs on the 7x7 arena, 50 episodes per point, no intervals. The
checkpoints behind these are gone, so they are not reproducible from this repo.

| Method | Success |
| ------ | ------- |
| Nav2 expert (demo source) | 89 % |
| BC + frame stacking + overrides | 62 % |
| BC alone | 52 % |
| PPO against frozen AIRL reward | 50 % |
| Adversarial AIRL | 46 % |

## BC training iterations (stage 4, 50-episode evals)

| Stage | Success | Mean reward | Mean length |
| ----- | ------- | ----------- | ----------- |
| Base BC | 28 % | -1271 | 264 |
| DAgger iter 1 | 50-60 % | | |
| DAgger iter 2 | 84 % | +1326 | 204 |
| + frame stacking | 80 % | +1207 | 168 |

These are 50-episode numbers on a checkpoint that no longer exists. The 300-
episode retrained figure in the main table (78.0 %) is the one to use.
