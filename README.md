# rl_on_wheels

Reinforcement learning for mobile robot navigation using ROS2 Humble + Ignition Fortress (Gazebo).

**Phase 1 — SAC** — obstacle avoidance + goal-reaching on TurtleBot3 Waffle Pi in a 6×6 m training room with 5 dynamic obstacles.  
Sensor input: 2D LiDAR (`/scan`, 360 rays → 36 normalised bins). No cameras.

---

## Roadmap

| Phase | Method | Status |
|-------|--------|--------|
| 1 | SAC (goal reaching, obstacle avoidance) | **current** |
| 2 | Domain Randomisation (sensor noise, friction, mass) | planned |
| 3 | Model-Based RL (world model + Dyna-style planning) | planned |
| 4 | Imitation Learning warm-start (behaviour cloning) | planned |
| 5 | ONNX export + real-robot deployment | planned |

---

## Repository layout

```
rl_on_wheels/
├── docker/
│   ├── Dockerfile               ROS2 Humble + Ignition Fortress + PyTorch
│   ├── docker-compose.yml       sim / train / eval services
│   ├── entrypoint.sh
│   └── requirements.txt
├── ros2_ws/src/tb3_rl_bridge/   C++ ROS2 package
│   ├── srv/                     GetObservation, Step, ResetEpisode services
│   ├── src/
│   │   ├── env_bridge_node.cpp  LiDAR + odom → obs, reward, done detection
│   │   └── reset_node.cpp       episode resets via gz-transport, goal marker
│   ├── worlds/tb3_training_room.sdf  Ignition world — 6×6 m room with waffle_pi
│   └── launch/bridge.launch.py  launches sim + bridge nodes
├── rl/
│   ├── envs/ros2_gym_env.py     gymnasium Env wrapper
│   ├── agents/sac_her.py        SAC build/load helpers
│   ├── train.py                 training entrypoint
│   └── eval.py                  evaluation / rollout
├── configs/sac_her.yaml         all hyperparameters + env settings
└── scripts/                     helper shell scripts
```

---

## Prerequisites

- **Docker >= 24** and **Docker Compose v2**
- **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **X11 display** for Gazebo GUI (native Linux desktop)

Verify GPU passthrough works:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Quickstart

### 1. Clone

```bash
git clone <repo-url> rl_on_wheels
cd rl_on_wheels
```

### 2. Build the Docker image

```bash
docker compose -f docker/docker-compose.yml build
```

This bakes in ROS2 Humble, Ignition Fortress, TurtleBot3 packages, PyTorch, and stable-baselines3.

### 3. Allow X11 access (for Gazebo GUI)

Run this on the **host** before starting the container:
```bash
xhost +local:root
```

### 4. Start the simulation container

```bash
docker run -it --rm \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/ros2_ws/src:/ros2_ws/src \
  -v $(pwd)/configs:/configs:ro \
  rl_on_wheels-sim bash
```

Inside the container, source and launch:
```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 launch tb3_rl_bridge bridge.launch.py
```

This starts:
- Ignition Fortress server (physics + sensors)
- Ignition GUI (3D viewer, headless-safe — GUI crash won't kill the sim)
- `ros_gz_bridge` — ROS2 ↔ Ignition topic bridge
- `env_bridge_node` — obs assembly, action execution, reward + done computation
- `reset_node` — random episode resets, obstacle management, goal sphere

To run headless (no GUI):
```bash
ros2 launch tb3_rl_bridge bridge.launch.py headless:=true
```

### 5. Rebuild after source changes

If you edit any C++ node:
```bash
cd /ros2_ws
colcon build --packages-select tb3_rl_bridge
source install/setup.bash
```

### 6. Run training (in a second terminal / second container)

```bash
docker run -it --rm \
  --network host \
  --ipc host \
  -v $(pwd)/rl:/rl:ro \
  -v $(pwd)/configs:/configs:ro \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/logs:/logs \
  rl_on_wheels-sim \
  python3 /rl/train.py --config /configs/sac_her.yaml
```

Resume from a checkpoint:
```bash
CHECKPOINT=checkpoints/sac_her_tb3_100000_steps \
  docker run ... python3 /rl/train.py --config /configs/sac_her.yaml \
    --checkpoint $CHECKPOINT
```

### 7. TensorBoard

```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0
# open http://localhost:6006
```

### 8. Evaluate a trained model

```bash
docker run -it --rm \
  --network host \
  -v $(pwd)/rl:/rl:ro \
  -v $(pwd)/configs:/configs:ro \
  -v $(pwd)/checkpoints:/checkpoints:ro \
  rl_on_wheels-sim \
  python3 /rl/eval.py --config /configs/sac_her.yaml \
    --checkpoint checkpoints/final_model
```

---

## Configuration

All hyperparameters live in [`configs/sac_her.yaml`](configs/sac_her.yaml).

| Key | Default | Description |
|-----|---------|-------------|
| `sac.total_timesteps` | 1 000 000 | training budget |
| `sac.learning_rate` | 1e-4 | Adam LR |
| `sac.gamma` | 0.99 | discount factor |
| `sac.ent_coef` | 0.1 | fixed entropy coefficient (prevents collapse) |
| `sac.learning_starts` | 3000 | random steps before first gradient update |
| `sac.gradient_steps` | 2 | gradient updates per env step |
| `env.lidar_bins` | 36 | downsampled LiDAR rays |
| `env.max_lidar_range` | 3.5 m | LiDAR clip range |
| `env.goal_tolerance` | 0.3 m | success radius |
| `env.collision_threshold` | 0.2 m | LiDAR min-range collision trigger |
| `env.max_episode_steps` | 600 | steps before truncation |
| `training.checkpoint_freq` | 10 000 | save every N steps |

---

## Architecture

```
Python (train.py)
    │  gym.step(action)
    ▼
TurtleBot3Env  ──── ROS2 DDS ────►  EnvBridgeNode (C++)
                                         │  /cmd_vel ──► Ignition
                                         │  /scan, /odom ◄── Ignition
                                         │  /goal_pose ◄── ResetNode
                                         │       └── gz-transport → set_pose, spawn/move obstacles
SAC.learn()
```

### ROS2 services

| Service | Direction | Description |
|---------|-----------|-------------|
| `/step` | Python → C++ | apply action, sleep step_duration, return (obs, reward, done) |
| `/get_observation` | Python → C++ | read current sensor state without acting |
| `/reset_episode` | Python → C++ | teleport robot + spawn new goal + reposition obstacles |

### Observation vector (41-dim)

```
[lidar_0 … lidar_35]   36 × normalised min-range ∈ [0, 1]   (robot frame)
[dist_norm]             distance to goal / max_lidar_range ∈ [0, 1]
[cos_goal]              cosine of goal heading in robot frame ∈ [−1, 1]
[sin_goal]              sine   of goal heading in robot frame ∈ [−1, 1]
[prev_lin_vel]          previous linear  velocity ∈ [0,    0.26] m/s
[prev_ang_vel]          previous angular velocity ∈ [−1.82, 1.82] rad/s
```

### Reward function

Paper §3.3 hybrid piecewise reward (Luo et al., Remote Sensing 16(12):2072):

```
r_cont = w_d × (d_prev − d_t) + w_θ × (θ_prev − θ_t)   (progress shaping)

r = +r_success  (10.0)   if dist < 0.3 m AND heading_error < 0.5 rad
r = −r_collision (10.0)  if min_lidar < 0.2 m
r = r_partial   (2.0, once per episode) + r_cont − r_step   if dist < 0.3 m (position only)
r = r_cont − r_step (0.005)                                  otherwise
```

Where `d_t` = distance to goal, `θ_t` = |heading error| (robot yaw vs goal direction),
`w_d = 1.0`, `w_θ = 0.5`. All constants are ROS2 parameters on `env_bridge_node`.

---

## License

MIT
