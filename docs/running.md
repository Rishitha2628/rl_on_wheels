# Running things

Everything runs inside the Docker image. Two terminals: one for the simulator,
one for whatever you are training or evaluating.

## Build

```bash
docker compose -f docker/docker-compose.yml build
xhost +local:root            # only needed if you want the Ignition GUI
```

## Simulator

```bash
docker compose -f docker/docker-compose.yml run -d --name sim sim sleep infinity
docker exec -it sim bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 headless:=true
```

Useful launch arguments:

| Argument | Default | Effect |
| -------- | ------- | ------ |
| `stage` | 4 | which arena, 1 to 5 |
| `headless` | false | no Ignition GUI |
| `fast` | false | unlimited real-time factor plus sim-clock stepping |
| `dynamic_obstacles` | true | set false to freeze the cylinders |
| `r_vlinear_scale` | 10.0 | forward-speed penalty scale |
| `r_vangular_weight` | 1.0 | turning penalty weight |

For several sims at once:

```bash
/scripts/launch_sim_fleet.sh -n 4 -s 4 --fast
/scripts/launch_sim_fleet.sh --stop
```

Instance `i` is on `ROS_DOMAIN_ID = 40 + i`.

## TD3

```bash
python3 /rl/train_td3.py --config /configs/td3.yaml
python3 /rl/train_td3.py --config /configs/td3.yaml --checkpoint /checkpoints/td3_tb3_214000_steps
```

With a budget grid, which checkpoints at set interaction counts so one run
gives the whole budget axis:

```bash
python3 /rl/sweep_td3.py --config /configs/td3_tune_a.yaml --seed 0 \
    --n-envs 4 --out-dir /checkpoints/tune/run --max-bridge-steps 640000
```

## Behavior cloning

Demo collection needs Nav2 running alongside the sim.

```bash
# terminal A
ros2 launch tb3_rl_bridge bridge.launch.py stage:=4 dynamic_obstacles:=true
# terminal B
ros2 launch tb3_nav2 nav2_bringup.launch.py stage:=4
# terminal C
ros2 bag record -o /demos/stage4_nav2 /scan /odom /robot_world_pose /goal_pose /cmd_vel
# terminal D
PYTHONPATH=/:$PYTHONPATH python3 /bc/run_episodes.py --episodes 300
```

Then build the dataset and train:

```bash
python3 /bc/build_dataset.py --bag /demos/stage4_nav2 --out /demos/stage4_bc.npz --goal-tolerance 0.45
python3 -m bc.train --config /configs/bc_stage4.yaml --dataset /demos/stage4_dagger_agg2.npz
python3 /bc/eval.py --checkpoint /checkpoints/bc_best.pt --episodes 300 --no-overrides
```

`--no-overrides` turns off the eval-time safety guards and P-controller. Use it
when comparing against the RL arms, which have no equivalent. Without the flag
you get the number the BC pipeline actually ships.

DAgger runs Nav2 in shadow mode, publishing to `/cmd_vel_expert` while BC drives:

```bash
python3 /bc/dagger_collect.py --checkpoint /checkpoints/bc_best.pt --out /demos/dagger_iter1.npz --episodes 200
python3 /bc/dataset_merge.py --inputs /demos/stage4_bc.npz /demos/dagger_iter1.npz --out /demos/stage4_dagger_agg1.npz
```

`bc/dagger_iterate.sh` wraps the loop.

## AIRL

Needs a BC-format NPZ first.

```bash
/scripts/prepare_airl.sh --out-dir /checkpoints/airl_stage4
```

That converts the demos, runs BC pretraining plus the adversarial loop, and
promotes the best-eval reward net over the final one. Then phase 2:

```bash
python3 /airl/sweep_airl.py --seed 0 --n-envs 4 \
    --reward-net  /checkpoints/airl_stage4/reward_net.pt \
    --init-policy /checkpoints/airl_stage4/policy_after_bc.zip \
    --out-dir /checkpoints/sweep/airl_seed0
```

## Evaluation and figures

```bash
python3 /rl/eval_budget.py --run-dir /checkpoints/sweep/td3_seed0 --episodes 100 --n-envs 4
python3 /scripts/analyze_crossover.py --sweep-dir /checkpoints/sweep \
    --tuned-dir /checkpoints/tune \
    --bc-results /checkpoints/sweep/bc_policy_alone.json \
    --bc-overrides-results /checkpoints/sweep/bc_with_overrides.json \
    --out-dir /logs/figures
```

`eval_budget.py` picks the algorithm and observation space from each run's
`run_meta.json`, so it handles both the TD3 and AIRL arms.

## Throughput checks

```bash
python3 /scripts/bench_throughput.py --config /configs/td3.yaml --steps 300
python3 /scripts/bench_fleet.py --config /configs/td3.yaml -n 4 --steps 120
python3 /scripts/equivalence_check.py record --arm rt --run a
python3 /scripts/equivalence_check.py compare
```

## TensorBoard

```bash
tensorboard --logdir /logs/tensorboard --host 0.0.0.0
```

## After editing C++

```bash
docker exec -it sim bash
touch /ros2_ws/src/tb3_rl_bridge/src/*.cpp
cd /ros2_ws && colcon build --packages-select tb3_rl_bridge
source install/setup.bash
```

The `touch` matters. The image's object files can carry a later mtime than a
host edit, in which case make skips the rebuild silently. Check with
`strings /ros2_ws/build/tb3_rl_bridge/env_bridge_node | grep <new-symbol>`.

`/ros2_ws/install` is a named Docker volume and is not re-initialised from a
fresh image, so it can hold stale build output. `/ros2_ws/build` comes from the
image and reverts whenever the container is recreated.
