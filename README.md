# rl_on_wheels

Reinforcement learning for mobile robot navigation on TurtleBot3 Waffle Pi
in ROS2 Humble + Ignition Fortress (Gazebo Sim).

Three agents are wired up — SAC (legacy), **TD3** (current focus), and PPO.
Sensor input is 2D LiDAR only (360 rays → 36 normalised bins). No cameras.

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
└── scripts/launch_sim.sh            in-container sim launcher
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

### 1. Build the Docker image (one-time)

```bash
docker compose -f docker/docker-compose.yml build
```

### 2. Allow X11 access (host side)

```bash
xhost +local:root
```

### 3. Terminal 1 — launch the simulator

Get a shell inside the sim container, then launch the bridge for your chosen stage:

```bash
docker compose -f docker/docker-compose.yml run --rm --name sim sim bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch tb3_rl_bridge bridge.launch.py stage:=4
```

### 4. Terminal 2 — run training

In a new host terminal, exec into the same container and run training:

```bash
docker exec -it sim bash
```

Inside the container:

```bash
# Fresh run
python3 /rl/train_td3.py --config /configs/td3.yaml

# Resume from a checkpoint
python3 /rl/train_td3.py --config /configs/td3.yaml \
    --checkpoint /checkpoints/td3_tb3_214000_steps
```

### 5. Rebuild after C++ changes

If you edit any `.cpp` under `tb3_rl_bridge/src/`, open a third terminal:

```bash
docker exec -it sim bash
cd /ros2_ws && colcon build --packages-select tb3_rl_bridge
source install/setup.bash
# then Ctrl-C the ros2 launch in Terminal 1 and re-run it
```

### 6. TensorBoard

From any container shell (host networking is on, so `localhost:6006` works from the host browser):

```bash
tensorboard --logdir /logs/tensorboard --host 0.0.0.0
```

### 7. Evaluate a checkpoint

In a container shell:

```bash
python3 /rl/eval.py --config /configs/td3.yaml \
    --checkpoint /checkpoints/td3_tb3_214000_steps
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
| `r_yaw = -abs(goal_angle)` | [−π, 0] | face the goal |
| `r_vangular = -ω²` | [−4, 0] | discourage spinning |
| `r_vlinear = -((v_max − v) × 10)²` | [−~5, 0] | encourage forward motion |
| `r_distance = 2·d₀/(d₀ + d) − 1` | [−1, 1] | shape progress toward goal |
| `r_obstacle = -20 if min_obs_dist < 0.22m else 0` | {0, −20} | avoid moving cylinders |

Wall proximity is handled via the collision termination (`-2000`),
not via `r_obstacle` — only moving cylinders trigger that penalty.

---

## License

MIT
