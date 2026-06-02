"""Launch a training stage.

Usage:
    ros2 launch tb3_rl_bridge bridge.launch.py stage:=4

Stages:
    1 — empty arena (5x5)
    2 — 4 static cylinders (5x5)
    3 — 4 cylinders with small oscillations (5x5)
    4 — inner walls + 2 dynamic cylinders (5x5) — canonical training stage
    5 — 7x7 maze, 8 inner walls + 6 dynamic cylinders — generalisation arena
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
from launch_ros.descriptions import ParameterValue


# Per-stage dynamic-obstacle configuration. Empty `active` → no
# dynamic_obstacle_node launched for that stage.
#
# motion_mode:
#   "keyframe" → KEYS_1..KEYS_6 large-path animations
#   "stage3"   → small-amplitude oscillations
# base_x / base_y: world positions (must match moving_obs_N <pose> in SDF)
STAGE_OBSTACLES = {
    "1":  {"active": "", "motion_mode": "keyframe",
           "base_x": [], "base_y": []},
    "2":  {"active": "", "motion_mode": "keyframe",
           "base_x": [], "base_y": []},
    # Stage 3 — 4 cylinders at (±1, ±1) doing small local oscillations.
    "3":  {"active": "1,2,3,4", "motion_mode": "stage3",
           "base_x": [-1.0, -1.0,  1.0, 1.0],
           "base_y": [-1.0,  1.0, -1.0, 1.0]},
    # Stage 4 — canonical training stage. 5x5 arena, 7 inner walls,
    # 2 keyframe cylinders.
    "4":  {"active": "1,2", "motion_mode": "keyframe",
           "base_x": [2.0, -2.0,  2.0, -2.0, -2.0, 2.0],
           "base_y": [2.0, -2.0, -2.0,  2.0,  0.0, 0.0]},
    # Stage 5 — 7x7 generalisation arena. 8 interior walls + 6 keyframe
    # cylinders. Bases scaled to ±2.8 so the cylinders cover the larger
    # arena evenly.
    "5":  {"active": "1,2,3,4,5,6", "motion_mode": "keyframe",
           "base_x": [2.8, -2.8,  2.8, -2.8, -2.8, 2.8],
           "base_y": [2.8, -2.8, -2.8,  2.8,  0.0, 0.0]},
}


def generate_launch_description():
    pkg_bridge      = get_package_share_directory("tb3_rl_bridge")
    pkg_ros_gz_sim  = get_package_share_directory("ros_gz_sim")

    stage           = LaunchConfiguration("stage",            default="4")
    use_sim_time    = LaunchConfiguration("use_sim_time",     default="true")
    collision_thr   = LaunchConfiguration("collision_threshold", default="0.13")
    max_lidar_range = LaunchConfiguration("max_lidar_range",  default="3.5")
    lidar_bins      = LaunchConfiguration("lidar_bins",       default="36")
    step_duration   = LaunchConfiguration("step_duration",    default="0.1")
    headless        = LaunchConfiguration("headless",         default="false")
    dynamic_obstacles = LaunchConfiguration("dynamic_obstacles", default="true")
    # "keyframe" = full per-obstacle paths (default for stages 4-5).
    # "stage3"   = small ±0.4 m local oscillations — easier for Nav2 to navigate around.
    motion_mode_override = LaunchConfiguration("motion_mode", default="")
    # >1 slows the cylinders proportionally. 3 = 1/3rd speed, etc.
    obstacle_time_scale = LaunchConfiguration("obstacle_time_scale", default="1.0")

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
                              description="Stage number 1-5"),
        DeclareLaunchArgument("use_sim_time",        default_value="true"),
        DeclareLaunchArgument("collision_threshold", default_value="0.13"),
        DeclareLaunchArgument("max_lidar_range",     default_value="3.5"),
        DeclareLaunchArgument("lidar_bins",          default_value="36"),
        DeclareLaunchArgument("step_duration",       default_value="0.1"),
        DeclareLaunchArgument("headless",            default_value="false"),
        DeclareLaunchArgument("dynamic_obstacles",   default_value="true",
                              description="set to false to keep moving cylinders stationary "
                                          "(useful for Nav2 demo collection)"),
        DeclareLaunchArgument("motion_mode",         default_value="",
                              description="override the per-stage motion_mode "
                                          "(empty=use STAGE_OBSTACLES default; "
                                          "'stage3' for small-amplitude oscillations)"),
        DeclareLaunchArgument("obstacle_time_scale", default_value="1.0",
                              description="multiplier for cylinder keyframe playback; "
                                          ">1 slows obstacles (try 3.0 for Nav2-friendly speed)"),

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
                # Stage 5 (7x7 arena) needs a longer lidar reach so BC sees a
                # comparable fraction of the world per observation. Hardware
                # max in tb3_stage5.sdf was also bumped to 6.0 m to match.
                "max_lidar_range": ParameterValue(
                    PythonExpression(["6.0 if '", stage, "' == '5' else 3.5"]),
                    value_type=float),
                "lidar_bins": lidar_bins,
                "collision_threshold": collision_thr,
                "goal_tolerance": 0.40,   # match Nav2's xy_goal_tolerance — demos never drove BC closer than this
                "step_duration": step_duration,
                "min_linear_vel":  0.0,    # no backward motion
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
                # Arena bounds: walls at ±2.425 (5x5 stages) or ±3.425 (stage 5),
                # clear inner area ≈ [-2.0, 2.0] or [-2.5, 2.5] respectively.
                "world_x_min": ParameterValue(
                    PythonExpression(["-2.5 if '", stage, "' == '5' else -2.0"]),
                    value_type=float),
                "world_x_max": ParameterValue(
                    PythonExpression(["2.5 if '", stage, "' == '5' else 2.0"]),
                    value_type=float),
                "world_y_min": ParameterValue(
                    PythonExpression(["-2.5 if '", stage, "' == '5' else -2.0"]),
                    value_type=float),
                "world_y_max": ParameterValue(
                    PythonExpression(["2.5 if '", stage, "' == '5' else 2.0"]),
                    value_type=float),
                "robot_min_goal_dist": 0.5,
                # Sphere radius matches goal_tolerance so the visible sphere
                # edge marks the trigger zone (robot touching sphere = success).
                "goal_sphere_radius": 0.20,
                "n_obstacles": 0,
                # Fixed robot spawn. spawn_x/y/theta below are used only
                # when reset_node's per-stage override doesn't fire (i.e.,
                # the stage isn't recognized). Stages 1-3 spawn at (0, 0)
                # and stages 4-5 at (-0.7, 0) — reset_node picks the
                # right one based on the `stage` param.
                "fixed_spawn": True,
                "spawn_x":    -0.7,
                "spawn_y":     0.0,
                "spawn_theta": 0.0,
                # Goal validity needs stage to know which inner-wall
                # rectangles to forbid.
                "arena_length": ParameterValue(
                    PythonExpression(["6.2 if '", stage, "' == '5' else 4.2"]),
                    value_type=float),
                "arena_width": ParameterValue(
                    PythonExpression(["6.2 if '", stage, "' == '5' else 4.2"]),
                    value_type=float),
                "stage":        stage,
            }],
        ),
    ]

    # ── dynamic_obstacle_node launched only for stages with moving obstacles ──
    # We register one node per stage and gate via IfCondition on both the stage
    # arg AND the dynamic_obstacles flag. dynamic_obstacles:=false keeps the
    # cylinders sitting at their SDF positions — useful for Nav2 demo runs.
    for stage_num, cfg in STAGE_OBSTACLES.items():
        if not cfg["active"]:
            continue
        nodes.append(Node(
            package="tb3_rl_bridge",
            executable="dynamic_obstacle_node",
            name=f"dynamic_obstacle_node_stage{stage_num}",
            output="screen",
            condition=IfCondition(PythonExpression([
                "'", stage, "' == '", stage_num,
                "' and '", dynamic_obstacles, "' == 'true'",
            ])),
            parameters=[{
                "use_sim_time": use_sim_time,
                "world_name": "empty",
                "update_rate_hz": 10.0,   # halved from 20 to reduce Ignition set_pose load during high-collision-rate stages
                # Cylinder SDF has a link-level <pose>0 0 0.25</pose> offset
                # already, so the MODEL-level teleport should use z=0 to keep
                # the cylinder sitting on the ground (not floating at 0.5).
                "obs_z": 0.0,
                "active_obstacles": cfg["active"],
                # If `motion_mode` launch arg is set, use it; otherwise the
                # per-stage default from STAGE_OBSTACLES.
                "motion_mode": PythonExpression([
                    "'", motion_mode_override, "' if '", motion_mode_override,
                    "' else '", cfg["motion_mode"], "'"]),
                "time_scale":       obstacle_time_scale,
                "obstacle_base_x":  cfg["base_x"],
                "obstacle_base_y":  cfg["base_y"],
            }],
        ))

    return LaunchDescription(nodes)
