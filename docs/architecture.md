# Architecture

Training runs in Python. The simulator and everything touching it runs in C++
ROS2 nodes. They talk over three ROS2 services.

```
Python (train_td3.py)
    |  gym.step(action)
    v
TurtleBot3Env  ---- ROS2 services ---->  env_bridge_node (C++)
                                             |  /cmd_vel ---> Ignition
                                             |  /scan, /odom <--- Ignition
                                             |  /goal_pose <--- reset_node
                                             |  /obstacle_poses <--- dynamic_obstacle_node
                                             +-- world-frame robot pose
                                                 <--- Ignition /dynamic_pose/info
```

## Services

| Service | Direction | Description |
| ------- | --------- | ----------- |
| `/step` | Python to C++ | apply action, wait `step_duration`, return (obs, reward, done) |
| `/get_observation` | Python to C++ | read current sensor state |
| `/reset_episode` | Python to C++ | teleport robot, spawn a new goal |

## Observation

`env_bridge_node` emits 42 floats:

```
[lidar_0 .. lidar_35]   36 min-range LiDAR bins, normalised to [0, 1]
[dist_norm]             distance to goal / max_lidar_range
[cos_goal_body]         cosine of goal heading in the robot frame
[sin_goal_body]         sine of goal heading in the robot frame
[goal_path_min]         min LiDAR in a +/-20 deg cone around the goal direction
[prev_lin_vel]          previous linear velocity
[prev_ang_vel]          previous angular velocity
```

BC and AIRL drop the last two and use 40. Feeding the previous action back in
creates an echo loop where the policy reproduces its last action regardless of
state. `bc/train.py` optionally stacks the last `frame_stack` observations
(default 4), giving a 160-dim input.

Actions are linear velocity in [0, 0.22] m/s and angular velocity in
[-2.0, 2.0] rad/s.

## Reward

Per-step shaping plus terminal bonuses. Defined in `env_bridge_node.cpp`, not in
the YAML configs (the reward weights that appear in the configs are unused).

```
reward = r_yaw + r_vangular + r_vlinear + r_distance + r_obstacle - 1
terminal: +2500 on goal reached, -2000 on collision
```

| Term | Range | Purpose |
| ---- | ----- | ------- |
| `r_yaw = -abs(goal_angle)` | [-pi, 0] | face the goal |
| `r_vangular = -w * omega^2` | [-4, 0] | discourage spinning |
| `r_vlinear = -((v_max - v) * scale)^2` | [-4.84, 0] | encourage forward motion |
| `r_distance = 2*d0/(d0 + d) - 1` | [-1, 1] | shape progress toward the goal |
| `r_obstacle = -20 if min_obs_dist < 0.22 m` | {0, -20} | avoid moving cylinders |

`scale` and `w` are the launch parameters `r_vlinear_scale` (default 10.0) and
`r_vangular_weight` (default 1.0). Walls are handled by the collision
termination, not by `r_obstacle`; only the moving cylinders trigger that.

## Simulation speed

The bridge blocks for `step_duration` of wall time on each step, which caps the
environment at 10 Hz no matter how fast Ignition could run. That is also the
rate a physical TurtleBot3 allows, so it is the default.

`fast:=true` swaps in a `*_fast.sdf` world (identical except
`real_time_factor` 1.0 -> 0) and switches the bridge to wait on the simulator
clock instead. Measured: 9.93 steps/s default, 58.2 accelerated on one
instance, 107.5 across four.

Two things had to be fixed before that was sound, both invisible at real-time
factor 1.0:

- `dynamic_obstacle_node` drove a simulated-time animation from a wall-clock
  timer. Under acceleration the cylinders were resampled far more coarsely per
  simulated second and jumped between distant points instead of sliding.
- `reset_node` measured its post-teleport settle delays in wall time, so
  episodes began at a different phase of the obstacle cycle in the two modes.

`scripts/equivalence_check.py` compares an open-loop rollout across modes. It
measures against the same-mode noise floor rather than against zero, because the
bridge is asynchronous and replaying the same actions twice in one mode already
diverges. Divergence came out at 1.11x the floor on the LiDAR block for a single
accelerated instance and 1.09x for one instance of a four-instance fleet.

## Parallel environments

`scripts/launch_sim_fleet.sh` starts N sims. Each needs two kinds of isolation:
`ROS_DOMAIN_ID` separates the DDS graph, and `IGN_PARTITION` separates Ignition's
own transport. Without the partition every instance teleports every other
instance's robot, because `reset_node` calls the Ignition services directly.

`rl/envs/vec_ros2_env.py` connects one `SubprocVecEnv` worker to each. It uses
`spawn`, not `fork` -- rclpy holds DDS participants and background threads that
do not survive a fork -- and sets `ROS_DOMAIN_ID` before `rclpy.init()`, which
reads it once at context creation.

Scaling is poor past four instances:

| Instances | Steps/s | Per instance | Efficiency |
| --------- | ------- | ------------ | ---------- |
| 1 | 58.2 | 58.2 | 1.00 |
| 2 | 77.9 | 39.0 | 0.67 |
| 4 | 107.5 | 26.9 | 0.46 |
| 6 | 107.2 | 17.9 | 0.31 |

Each instance costs about 2.83 CPU cores, so a 16-thread machine saturates
before memory does.

## Update-to-data ratio

SB3 collects `n_envs` transitions per round. Leaving `gradient_steps` at 1 means
one update per `n_envs` transitions, against one per transition in the
single-env reference -- a different algorithm rather than a faster one.
`td3_agent.build()` takes `gradient_steps_override` and `rl/sweep_td3.py` sets
it to `config_value * n_envs`. Check it in the log: `n_updates` should track
`total_timesteps - learning_starts` closely.

The on-policy equivalent is in `airl/sweep_airl.py`: PPO updates every
`n_steps * n_envs` env steps, so `n_steps` is divided by `n_envs`.

## Known issue

The stage worlds set the LiDAR `<update_rate>` to 5 while the environment steps
at 10 Hz, so every second observation reuses the previous scan. All methods
share it, so the comparison is internally consistent. Fixing it changes the task
and would need everything re-baselined.
