# rl_on_wheels

https://github.com/user-attachments/assets/d807a534-3d17-4f3a-8efa-861af5b8f4ba

Goal-conditioned navigation for a TurtleBot3 Waffle Pi in ROS2 Humble and
Ignition Fortress. A 2D LiDAR (360 rays reduced to 36 bins) plus the goal
position in the robot frame, no cameras. Three ways of learning the policy are
implemented against the same environment so they can be compared directly:

- **Behavior cloning + DAgger** on demonstrations from Nav2
- **AIRL**, recovering a reward from those demonstrations and then optimising
  PPO against it
- **TD3** on the environment's own reward

## Results

Stage 4 (5x5 m arena, 7 inner walls, 2 moving cylinders). Deterministic
evaluation, success is reaching the goal within 0.40 m, Wilson 95% intervals.

| Method | Success | 95% CI | Episodes |
| ------ | ------- | ------ | -------- |
| BC + safety overrides | 78.0 % | [73.0, 82.3] | 300 |
| TD3, lr 3e-4 / sigma 0.1 (3 seeds) | 68.0 % | [63, 73] | 300 |
| BC policy alone | 63.7 % | [58.1, 68.9] | 300 |
| TD3, lr 1e-4 / sigma 0.05 (3 seeds) | 62.7 % | [57, 68] | 300 |
| AIRL phase 2, best point | 23.7 % | [19, 29] | 300 |

The two BC rows are the same network. The difference is the eval-time wrapper
in `bc/eval.py` -- two clearance overrides and a P-controller for the final
approach -- which is worth 14.3 points.

No RL configuration I tried gets past about 69 %. Two things that looked like
they might explain the gap turned out not to:

- The shipped TD3 hyperparameters (lr 1e-4, OU sigma 0.05) were meant for
  fine-tuning an already-converged policy, per the comments in
  `configs/td3.yaml`. Using the SB3 defaults instead gains about 5 points
  across 3 seeds, which is not enough.
- The reward penalises slowing and turning more heavily than it penalises
  collision. Softening that by 4x and 16x changes nothing (68 -> 69 -> 69).

Full tables, the budget sweep, and the per-seed numbers are in
[docs/results.md](docs/results.md).

## Setup

Needs Docker with the NVIDIA container toolkit, and an X11 display if you want
the Ignition GUI.

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run -d --name sim sim sleep infinity
```

Launch the simulator in one shell:

```bash
docker exec -it sim bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true
```

and train in another:

```bash
docker exec -it sim bash
python3 /rl/train_td3.py --config /configs/td3.yaml
```

Per-method commands (BC, DAgger, AIRL), the evaluation scripts, and the
parallel-simulator setup are in [docs/running.md](docs/running.md).

## Layout

```
bc/                       behavior cloning and DAgger
airl/                     AIRL: reward recovery, then PPO against it
rl/                       TD3 / PPO / SAC, the gym wrapper, budget sweeps
configs/                  hyperparameters and env settings
scripts/                  sim launchers, benchmarks, evaluation harnesses
docker/                   ROS2 Humble + Ignition Fortress image
ros2_ws/src/tb3_rl_bridge/   C++ bridge: obs, reward, resets, moving obstacles
ros2_ws/src/tb3_nav2/     Nav2 stack, used to collect the demonstrations
docs/                     results, architecture, run instructions
```

Demonstrations and checkpoints are not in the repo. The ros2 bags run to a few
GB and three are past GitHub's file size limit.

## Notes

The bridge sleeps for the control period on every step, so the environment runs
at 10 Hz regardless of machine speed -- the same rate the real robot allows.
`fast:=true` removes that cap for training; `scripts/equivalence_check.py`
checks the two modes produce the same trajectories.

`action_repeat` is 4, so a configured 1M timesteps is 4M environment steps.
Worth remembering when reading any budget number.

[docs/architecture.md](docs/architecture.md) covers the bridge design, the
observation and reward definitions, and the parallel-simulator setup.

## License

MIT
