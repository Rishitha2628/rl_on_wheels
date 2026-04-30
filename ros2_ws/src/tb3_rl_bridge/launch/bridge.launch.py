import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                             IncludeLaunchDescription, RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bridge = get_package_share_directory("tb3_rl_bridge")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    world_file = os.path.join(pkg_bridge, "worlds", "tb3_empty.sdf")

    use_sim_time        = LaunchConfiguration("use_sim_time",        default="true")
    collision_threshold = LaunchConfiguration("collision_threshold",  default="0.2")
    max_lidar_range     = LaunchConfiguration("max_lidar_range",     default="3.5")
    lidar_bins          = LaunchConfiguration("lidar_bins",          default="36")
    step_duration       = LaunchConfiguration("step_duration",       default="0.1")
    headless            = LaunchConfiguration("headless",             default="false")

    # ── Ignition server (always runs, physics + sensors + gz-transport) ──────
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-r -s " + world_file}.items(),
    )

    # ── Ignition GUI (optional — crash does NOT kill the simulation) ─────────
    # Connects to the already-running server via gz-transport discovery.
    gz_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-g"],
        output="screen",
        condition=UnlessCondition(headless),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time",        default_value="true"),
        DeclareLaunchArgument("collision_threshold",  default_value="0.2"),
        DeclareLaunchArgument("max_lidar_range",     default_value="3.5"),
        DeclareLaunchArgument("lidar_bins",          default_value="36"),
        DeclareLaunchArgument("step_duration",       default_value="0.1"),
        DeclareLaunchArgument("headless",            default_value="false",
                              description="true = skip GUI entirely"),

        # Server — required (simulation can't run without it)
        gz_server,

        # GUI — optional (if OpenGL fails, server + training keep running)
        gz_gui,

        # ── ROS2 ↔ Ignition topic bridge ─────────────────────────────────────
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

        # ── RL bridge nodes ───────────────────────────────────────────────────
        Node(
            package="tb3_rl_bridge",
            executable="env_bridge_node",
            name="env_bridge_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "max_lidar_range": max_lidar_range,
                "lidar_bins": lidar_bins,
                "collision_threshold": collision_threshold,
                "goal_tolerance": 0.3,
                "step_duration": step_duration,
                "max_linear_vel": 0.26,
                "max_angular_vel": 0.5,
            }],
        ),

        Node(
            package="tb3_rl_bridge",
            executable="reward_node",
            name="reward_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "collision_threshold": collision_threshold,
                "max_lidar_range": max_lidar_range,
                "goal_tolerance": 0.3,
                "collision_penalty": -1.0,
                "goal_reward": 1.0,
                "time_penalty": -0.005,
                "distance_reward_scale": 1.0,
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
                "world_x_min": -2.0,
                "world_x_max":  2.0,
                "world_y_min": -2.0,
                "world_y_max":  2.0,
                "robot_min_goal_dist": 0.5,
                "goal_sphere_radius": 0.1,
                "n_obstacles": 1,
            }],
        ),
    ])
