"""Launch a drlnav-style training stage.

Usage:
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4

Stages (matching tomasvr/turtlebot3_drlnav):
    1   — empty arena
    2,3 — 4 static cylinders
    4   — inner walls + 2 moving obstacles  (canonical training stage)
    5   — inner walls + 6 moving obstacles
    6   — 6 moving obstacles (no inner walls)
    7-10 — inner walls + 2 moving obstacles (variants for eval)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                             IncludeLaunchDescription)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Per-stage dynamic-obstacle configuration. Empty `active` → no
# dynamic_obstacle_node launched for that stage.
#
# motion_mode:
#   "drlnav" → KEYS_1..KEYS_6 from drlnav obstacle{1..6}.cc (large paths)
#   "stage3" → small-amplitude oscillations (libobstacles.so equivalent)
# base_x / base_y: world positions (must match moving_obs_N <pose> in SDF)
STAGE_OBSTACLES = {
    "1":  {"active": "", "motion_mode": "drlnav",
           "base_x": [], "base_y": []},
    "2":  {"active": "", "motion_mode": "drlnav",
           "base_x": [], "base_y": []},
    # Stage 3 — 4 cylinders at (±1, ±1) doing small local oscillations.
    "3":  {"active": "1,2,3,4", "motion_mode": "stage3",
           "base_x": [-1.0, -1.0,  1.0, 1.0],
           "base_y": [-1.0,  1.0, -1.0, 1.0]},
    # Stages 4-10 — drlnav-exact keyframe paths.
    "4":  {"active": "1,2", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "5":  {"active": "1,2,3,4,5,6", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "6":  {"active": "1,2,3,4,5,6", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "7":  {"active": "1,2", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "8":  {"active": "1,2", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "9":  {"active": "1,2", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    "10": {"active": "1,2", "motion_mode": "drlnav",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
}


def generate_launch_description():
    pkg_bridge      = get_package_share_directory("tb3_rl_bridge")
    pkg_ros_gz_sim  = get_package_share_directory("ros_gz_sim")

    stage           = LaunchConfiguration("stage",            default="4")
    use_sim_time    = LaunchConfiguration("use_sim_time",     default="true")
    collision_thr   = LaunchConfiguration("collision_threshold", default="0.13")  # drlnav THRESHOLD_COLLISION
    max_lidar_range = LaunchConfiguration("max_lidar_range",  default="3.5")
    lidar_bins      = LaunchConfiguration("lidar_bins",       default="36")
    step_duration   = LaunchConfiguration("step_duration",    default="0.1")
    headless        = LaunchConfiguration("headless",         default="false")

    # World file resolved at launch time via substitution
    world_file = PythonExpression([
        "'", os.path.join(pkg_bridge, "worlds"), "/tb3_stage' + '",
        stage, "' + '.sdf'"
    ])

    # ── Ignition server ───────────────────────────────────────────────────────
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r -s ", world_file]}.items(),
    )

    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-g"],
        output="screen",
        condition=UnlessCondition(headless),
    )

    nodes = [
        DeclareLaunchArgument("stage",               default_value="4",
                              description="drlnav stage number 1-10"),
        DeclareLaunchArgument("use_sim_time",        default_value="true"),
        DeclareLaunchArgument("collision_threshold", default_value="0.13"),
        DeclareLaunchArgument("max_lidar_range",     default_value="3.5"),
        DeclareLaunchArgument("lidar_bins",          default_value="36"),
        DeclareLaunchArgument("step_duration",       default_value="0.1"),
        DeclareLaunchArgument("headless",            default_value="false"),

        gz_server,
        gz_gui,

        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_bridge",
            output="screen",
            arguments=[
                "/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                "/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
                "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
                "/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
                "/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
                "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        Node(
            package="tb3_rl_bridge",
            executable="env_bridge_node",
            name="env_bridge_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "max_lidar_range": max_lidar_range,
                "lidar_bins": lidar_bins,
                "collision_threshold": collision_thr,
                "goal_tolerance": 0.20,                # drlnav THREHSOLD_GOAL
                "step_duration": step_duration,
                "min_linear_vel":  0.0,    # drlnav ENABLE_BACKWARD=False
                "max_linear_vel":  0.22,
                "max_angular_vel": 2.0,
                "clearance_threshold": 0.8,
            }],
        ),

        Node(
            package="tb3_rl_bridge",
            executable="reset_node",
            name="reset_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "tb3_model": "waffle_pi",
                "world_name": "empty",
                # drlnav arena bounds: walls at ±2.425, clear inner area ≈ [-2.0, 2.0]
                "world_x_min": -2.0,
                "world_x_max":  2.0,
                "world_y_min": -2.0,
                "world_y_max":  2.0,
                "robot_min_goal_dist": 0.5,
                # Sphere radius matches goal_tolerance so the visible sphere
                # edge marks the trigger zone (robot touching sphere = success).
                "goal_sphere_radius": 0.20,
                "n_obstacles": 0,
                # drlnav-style fixed robot spawn. spawn_x/y/theta below are
                # used only when reset_node's per-stage override doesn't fire
                # (i.e., the stage isn't recognized). drlnav uses (0,0) for
                # stages 1-3 and (-0.7, 0) for stages 4-10 — reset_node
                # picks the right one based on the `stage` param.
                "fixed_spawn": True,
                "spawn_x":    -0.7,
                "spawn_y":     0.0,
                "spawn_theta": 0.0,
                # drlnav goal validity — needs stage to know which
                # inner-wall rectangles to forbid.
                "arena_length": 4.2,
                "arena_width":  4.2,
                "stage":        stage,
            }],
        ),
    ]

    # ── dynamic_obstacle_node launched only for stages with moving obstacles ──
    # We register one node per stage and gate via IfCondition on the stage arg.
    for stage_num, cfg in STAGE_OBSTACLES.items():
        if not cfg["active"]:
            continue
        nodes.append(Node(
            package="tb3_rl_bridge",
            executable="dynamic_obstacle_node",
            name=f"dynamic_obstacle_node_stage{stage_num}",
            output="screen",
            condition=IfCondition(
                PythonExpression(["'", stage, "' == '", stage_num, "'"])),
            parameters=[{
                "use_sim_time": use_sim_time,
                "world_name": "empty",
                "update_rate_hz": 10.0,   # halved from 20 to reduce Ignition set_pose load during high-collision-rate stages
                # Cylinder SDF has a link-level <pose>0 0 0.25</pose> offset
                # already, so the MODEL-level teleport should use z=0 to keep
                # the cylinder sitting on the ground (not floating at 0.5).
                "obs_z": 0.0,
                "active_obstacles": cfg["active"],
                "motion_mode":      cfg["motion_mode"],
                "obstacle_base_x":  cfg["base_x"],
                "obstacle_base_y":  cfg["base_y"],
            }],
        ))

    return LaunchDescription(nodes)
