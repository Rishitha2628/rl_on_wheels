# rl_on_wheels

https://github.com/user-attachments/assets/d807a534-3d17-4f3a-8efa-861af5b8f4ba

Goal-conditioned navigation for a TurtleBot3 Waffle Pi in ROS2 Humble and
Ignition Fortress, among moving cylindrical obstacles. A 2D LiDAR (360 rays
reduced to 36 bins) plus the goal position in the robot frame, no cameras.
Three ways of learning the policy are implemented against the same environment
so they can be compared directly:

- **Behaviour cloning + DAgger** on demonstrations from Nav2
- **TD3** on the environment's own reward
- **AIRL**, recovering a reward from those demonstrations and then optimising
  PPO against it

The project started as a comparison of those three. It turned into a study of
how much a reported difference between design choices actually survives the
run-to-run noise of the stack underneath them.

## Results

7x7 m arena, 6 moving cylinders. Deterministic evaluation, 300 episodes,
success is reaching the goal within 0.40 m before the 450-step limit.

| Method | Success |
| ------ | ------- |
| BC + DAgger + eval wrapper | 69.7 % |
| TD3 | 63.3 % |
| BC + DAgger | 61.3 % |
| AIRL | 50.7 % |

The top two BC rows are the same network; the difference is the eval-time
wrapper in `bc/eval.py` -- two clearance overrides and a P-controller for the
final approach.

### What the noise floor does to all of that

Two evaluations of the *same* policy over 300 episodes each can differ by up to
**7.9 points**. Two of the gaps in the table above are smaller than that.

Re-running an identical 2x2 configuration grid (frames x control rate) changed
both which configuration won and the best-minus-worst gap: on the 5x5 arena it
grew from 4.3 to 8.0 points, on the 7x7 arena it fell from 11.7 to 9.3. Nothing
differed between the two runs. A grid's best-minus-worst gap is a selection
statistic, not an effect size, so individual comparisons are tested directly
instead, paired on identical episodes and Holm-corrected.

### Design choices, tested one at a time

- **Control rate** has no detectable effect once each policy is trained at the
  rate it is evaluated at: 55.8 % at 2.5 Hz against 57.1 % at 10 Hz over four
  seeds each, a −1.3 point difference (p = 0.55).
- **The eval wrapper** is the most reliable effect in the project, worth about
  9 points and significant after correction. It is hand-written engineering,
  not learning.
- **Frame stacking** gives about +5.5 points on the 7x7 arena across four seeds
  and nothing on the 5x5. It is not explained by seeing the obstacles move (the
  gain survives with them frozen) or by averaging sensor noise (it shrinks
  under every degradation tested). Policies trained with history barely use it
  at inference: freezing the history of a trained 4-frame policy costs only
  1.7 points.
- **At a fixed transition budget**, subsampling all 666 demonstrations beats
  using 168 complete ones by a wide margin -- episode variety matters more than
  sampling density.

### Transfer

Neither method transfers. In an unseen floor plan both fall from about 62 % to
below 20 %, a drop of 43-44 points, and neither is better than the other. A
robot that simply drives straight at the goal succeeds on 14.4 % there, so
learning adds only 3-6 points. Nav2, given the map, holds 51.3 % in the same
arena -- the task is solvable, just not from 36 LiDAR readings.

Full tables and per-seed numbers are in [docs/results.md](docs/results.md).

## Measurement protocol

Three conventions are enforced in code rather than by habit, because without
them none of the numbers above are comparable:

1. **Goal sequences are pinned.** Every evaluation draws from the same seeded
   sequence, so two policies face the same 300 tasks in the same order and can
   be compared episode by episode. `bc/eval.py` rewinds it per run.
2. **Evaluation is deterministic.** No action noise, no exploration.
3. **Success is decided in the C++ bridge**, so the criterion is identical for
   every method.

Where seeds are involved the unit of analysis is the seed, not the episode.

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
bc/                       behaviour cloning, DAgger, history masking
airl/                     AIRL: reward recovery, then PPO against it
rl/                       TD3 / PPO / SAC, the gym wrapper, budget sweeps
configs/                  hyperparameters and env settings, one per cell
scripts/                  sim launchers, experiments, benchmarks, analysis
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
checks the two modes produce the same trajectories. Four parallel instances
reach about 107 steps/s against 9.93 real-time, and each needs its own
`ROS_DOMAIN_ID` *and* `IGN_PARTITION` -- setting only the first leaves them
sharing a simulator bus.

`action_repeat` is 4, so a configured 1M timesteps is 4M environment steps.
Worth remembering when reading any budget number.

The BC observation is 40-dim: 36 LiDAR bins, goal distance, cos/sin of goal
bearing, and the minimum range in a cone toward the goal. The bridge emits 42 --
the previous linear and angular velocity are dropped on purpose, because given
them the policy learns to echo its own last command.

[docs/architecture.md](docs/architecture.md) covers the bridge design, the
observation and reward definitions, and the parallel-simulator setup.

## License

MIT
