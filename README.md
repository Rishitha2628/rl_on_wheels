# rl_on_wheels

Reinforcement learning for mobile robot navigation on TurtleBot3 Waffle Pi
in ROS2 Humble + Ignition Fortress (Gazebo Sim).

Three agents are wired up — SAC (legacy), **TD3** (current focus), and PPO —
plus a 10-stage progressive curriculum from an empty arena up through inner
walls and moving obstacles. Sensor input is 2D LiDAR only (360 rays → 36
normalised bins). No cameras.

---

## Stage curriculum

| Stage | World layout |
|-------|--------------|
| 1 | empty 5×5 m arena |
| 2 | 4 static cylinders at (±1, ±1) |
| 3 | 4 cylinders doing small-amplitude oscillations |
| 4 | inner walls + 2 moving cylinders *(canonical training stage)* |
| 5 | inner walls + 6 moving cylinders |
| 6 | 6 moving cylinders, no inner walls |
| 7–10 | inner walls + 2 moving cylinders (eval variants, different goal sets) |

Robot spawns at `(0, 0)` for stages 1–3 and `(-0.7, 0)` for stages 4–10.
Goal positions are stage-dependent — random sampling on the easier stages,
fixed goal lists on stages 4, 5, 7, 8, 9.

---

## Repository layout

```
rl_on_wheels/
├── docker/                          ROS2 Humble + Ignition Fortress image
├── ros2_ws/src/tb3_rl_bridge/       C++ ROS2 package
│   ├── srv/                         GetObservation, Step, ResetEpisode
│   ├── src/
│   │   ├── env_bridge_node.cpp      obs assembly, reward, done detection
│   │   ├── reset_node.cpp           teleport + goal sampling per stage
│   │   └── dynamic_obstacle_node.cpp keyframe-driven moving cylinders
│   ├── worlds/tb3_stage{1..10}.sdf  per-stage Ignition world files
│   ├── tools/gen_stages.py          regenerates stage SDFs from template
│   └── launch/bridge.launch.py      starts sim + bridge nodes
├── rl/
│   ├── envs/ros2_gym_env.py         gymnasium Env wrapper
│   ├── agents/{sac_her,td3,ppo}_agent.py  per-method build/load helpers
│   ├── train.py / train_td3.py / train_ppo.py
│   └── eval.py
├── configs/{sac_her,td3,ppo}.yaml   hyperparameters + env settings
└── scripts/
    ├── launch_sim.sh                in-container sim launcher
    └── run_stage.sh                 host-side helper: `./run_stage.sh <N>`
```

---

## Prerequisites

- **Docker ≥ 24** and **Docker Compose v2**
- **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **X11 display** for Ignition GUI (native Linux desktop)

Verify GPU passthrough:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Quickstart

### 1. Build the image
```bash
docker compose -f docker/docker-compose.yml build
```

### 2. Allow X11 (host side)
```bash
xhost +local:root
```

### 3. Train TD3 on a chosen stage

The simplest path uses `scripts/run_stage.sh`, which brings up the sim and
the trainer together:

```bash
./scripts/run_stage.sh 4                                # fresh
./scripts/run_stage.sh 4 /checkpoints/td3_tb3_100000_steps  # resume
```

To run sim and training in separate terminals (useful for debugging):

```bash
# terminal 1: sim
STAGE=4 docker compose -f docker/docker-compose.yml up sim

# terminal 2: training
docker compose -f docker/docker-compose.yml run --rm train_td3
```

Headless sim (no Ignition GUI):
```bash
HEADLESS=1 STAGE=4 docker compose -f docker/docker-compose.yml up sim
```

### 4. TensorBoard
```bash
# inside the train_td3 container, port 6007 is mapped (6006 is SAC's)
open http://localhost:6007
```

### 5. Rebuild after C++ changes
If you edit any `.cpp` under `tb3_rl_bridge/src/`, rebuild inside the sim
container:
```bash
docker exec -it <sim_container> bash
cd /ros2_ws && colcon build --packages-select tb3_rl_bridge
source install/setup.bash
# then relaunch
```

### 6. Evaluate a checkpoint
```bash
CHECKPOINT=checkpoints/td3_tb3_214000_steps \
  docker compose -f docker/docker-compose.yml run --rm eval
```

---

## Switching methods

| Method | Config | Train script | TensorBoard port |
|--------|--------|--------------|-------|
| SAC (legacy) | `configs/sac_her.yaml` | `rl/train.py` | 6006 |
| TD3 | `configs/td3.yaml` | `rl/train_td3.py` | 6007 |
| PPO | `configs/ppo.yaml` | `rl/train_ppo.py` | — |

---

## Architecture

```
Python (train_td3.py)
    │  gym.step(action)
    ▼
TurtleBot3Env  ──── ROS2 services ────►  env_bridge_node (C++)
                                              │  /cmd_vel ──► Ignition
                                              │  /scan, /odom ◄── Ignition
                                              │  /goal_pose  ◄── reset_node
                                              │  /obstacle_poses ◄── dynamic_obstacle_node
                                              └── world-frame robot pose
                                                  ◄── Ignition /dynamic_pose/info
```

### ROS2 services

| Service | Direction | Description |
|---------|-----------|-------------|
| `/step` | Python → C++ | apply action, wait `step_duration`, return (obs, reward, done) |
| `/get_observation` | Python → C++ | read current sensor state |
| `/reset_episode` | Python → C++ | teleport robot + spawn new goal |

### Observation vector (40-dim)

```
[lidar_0 … lidar_35]    36 × normalised LiDAR ∈ [0, 1]    (robot frame)
[dist_norm]              distance to goal / max_lidar_range
[cos_goal_body]          cosine of goal heading in robot frame
[sin_goal_body]          sine   of goal heading in robot frame
[prev_lin_vel]           previous linear  velocity
```

### Reward (current TD3 setup)

Per-step shaping plus large terminal bonuses:

```
reward = r_yaw + r_vangular + r_vlinear + r_distance + r_obstacle − 1

terminal: +2500 on goal reached, −2000 on collision
```

| Term | Range | Purpose |
|------|-------|---------|
| `r_yaw = -|goal_angle|` | [−π, 0] | face the goal |
| `r_vangular = -ω²` | [−4, 0] | discourage spinning |
| `r_vlinear = -((v_max − v) × 10)²` | [−~5, 0] | encourage forward motion |
| `r_distance = 2·d₀/(d₀ + d) − 1` | [−1, 1] | shape progress toward goal |
| `r_obstacle = -20 if min_obs_dist < 0.22m else 0` | {0, −20} | avoid moving cylinders |

Wall proximity is handled via the collision termination (`-2000`),
not via `r_obstacle` — only moving cylinders trigger that penalty.

---

## Configuration highlights (TD3)

All hyperparameters live in [`configs/td3.yaml`](configs/td3.yaml).

| Key | Default | Description |
|-----|---------|-------------|
| `td3.learning_rate` | 1e-4 | Adam LR (3e-4 default; lower for fine-tune) |
| `td3.buffer_size` | 1 000 000 | replay buffer capacity |
| `td3.batch_size` | 128 | minibatch size |
| `td3.tau` | 0.003 | target network soft-update rate |
| `td3.policy_delay` | 2 | actor update every N critic updates |
| `td3.action_noise.sigma` | 0.05 | OU exploration noise σ |
| `td3.net_arch` | [512, 512] | hidden layers |
| `td3.frame_stack` | 1 | number of stacked observation frames |
| `td3.action_repeat` | 4 | hold each action for N sim steps |
| `env.max_episode_steps` | 450 | timeout |
| `env.collision_threshold` | 0.13 m | LiDAR min-range collision trigger |
| `env.goal_tolerance` | 0.20 m | success radius |
| `training.checkpoint_freq` | 2 000 | save every N steps |

---

## License

MIT
