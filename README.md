# rl_on_wheels

Reinforcement learning for mobile robot navigation using ROS2 Humble + Ignition Fortress (Gazebo).

**Phase 1 — SAC + HER** — goal-reaching on TurtleBot3 Waffle Pi in an empty world.  
Sensor input: 2D LiDAR (`/scan`, 360 rays → 36 normalised bins). No cameras.

---

## Roadmap

| Phase | Method | Status |
|-------|--------|--------|
| 1 | SAC + HER (goal reaching, empty world) | **current** |
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
│   │   ├── env_bridge_node.cpp  LiDAR + odom → obs, executes actions
│   │   ├── reward_node.cpp      potential-based reward, done detection
│   │   └── reset_node.cpp       episode resets via gz-transport, goal marker
│   ├── worlds/tb3_empty.sdf     Ignition world with waffle_pi model
│   └── launch/bridge.launch.py  launches sim + bridge nodes
├── rl/
│   ├── envs/ros2_gym_env.py     gymnasium GoalEnv wrapper
│   ├── networks/actor_critic.py LidarMlpExtractor (robot-frame goal input)
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
- `env_bridge_node` — obs assembly + action execution
- `reward_node` — reward + done signal
- `reset_node` — random episode resets + goal sphere management

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
  docker run ... python3 /rl/train.py --config /configs/sac_her.yaml
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
| `sac.gamma` | 0.95 | discount factor |
| `sac.ent_coef` | 0.1 | fixed entropy coefficient (prevents collapse) |
| `sac.learning_starts` | 5000 | random steps before first gradient update |
| `sac.gradient_steps` | 2 | gradient updates per env step |
| `her.n_sampled_goal` | 4 | HER relabelled goals per real transition |
| `her.goal_selection_strategy` | future | HER strategy |
| `env.lidar_bins` | 36 | downsampled LiDAR rays |
| `env.max_lidar_range` | 3.5 m | LiDAR clip range |
| `env.goal_tolerance` | 0.35 m | success radius |
| `env.collision_threshold` | 0.2 m | LiDAR min-range collision trigger |
| `env.max_episode_steps` | 800 | steps before truncation |
| `env.reward_type` | dense | `dense` or `sparse` |
| `training.checkpoint_freq` | 10 000 | save every N steps |

---

## Architecture

```
Python (train.py)
    │  gym.step(action)
    ▼
TurtleBot3Env  ──── ROS2 DDS ────►  EnvBridgeNode (C++)
    │                                    │  /cmd_vel ──► Ignition
    │  compute_reward()                  │  /scan, /odom ◄── Ignition
    │  (pure Python, for HER)            │  /reward, /episode_done ◄── RewardNode
    ▼                                    │  /reset_episode ──► ResetNode
HerReplayBuffer                          │       └── gz-transport → set_pose, spawn/remove
    │
SAC.learn()
```

### ROS2 services

| Service | Direction | Description |
|---------|-----------|-------------|
| `/step` | Python → C++ | apply action, sleep step_duration, return (obs, reward, done) |
| `/get_observation` | Python → C++ | read current sensor state without acting |
| `/reset_episode` | Python → C++ | teleport robot + spawn new goal |

### Observation vector (40-dim)

```
[lidar_0 … lidar_35]   36 × normalised range ∈ [0, 1]   (robot frame)
[lin_vel]               linear  velocity ∈ [−0.26,  0.26] m/s
[ang_vel]               angular velocity ∈ [−0.50,  0.50] rad/s
[cos_yaw]               cosine of robot heading ∈ [−1, 1]
[sin_yaw]               sine   of robot heading ∈ [−1, 1]
```

`achieved_goal` = `[robot_x, robot_y]` in odom frame  
`desired_goal`  = `[goal_x,  goal_y]`  in odom frame

The feature extractor (`LidarMlpExtractor`) rotates the world-frame relative goal
into robot frame using yaw before passing it to the MLP, so lidar and goal share
the same coordinate system.

### Reward function

```
r = −clip(dist, 3.5) / 3.5 − 0.005        (dense, every step)
r = +1.0                                    (goal reached: dist < 0.35 m)
r = −1.0                                    (collision:    min_lidar < 0.2 m)
```

Stateless potential-based reward — compatible with HER offline relabelling.

---

## Coordinate frames

The robot is teleported each episode via Ignition's `set_pose` service. Since
the DiffDrive plugin does not reset odometry on teleport, `reset_node` converts
the goal from world frame to odom frame at each reset so that all position
arithmetic in `reward_node` and `env_bridge_node` is internally consistent.

---

## License

MIT
