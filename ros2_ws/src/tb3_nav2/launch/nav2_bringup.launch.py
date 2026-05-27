"""Bring up Nav2 for the chosen training stage.

Run AFTER the sim is already up (bridge.launch.py with the same stage):
    ros2 launch tb3_nav2 nav2_bringup.launch.py stage:=4

Components started:
  - pose_publisher_node : publishes map→odom from Ignition ground-truth
  - map_server          : serves the static occupancy grid for the stage
  - planner_server      : global path planning (NavfnPlanner)
  - controller_server   : local control (Regulated Pure Pursuit)
  - behavior_server     : recovery behaviors (spin, backup, wait)
  - bt_navigator        : behavior tree for NavigateToPose action
  - lifecycle_manager   : configures + activates all of the above
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("tb3_nav2")

    stage         = LaunchConfiguration("stage",        default="4")
    use_sim_time  = LaunchConfiguration("use_sim_time", default="true")
    params_file   = LaunchConfiguration(
        "params_file",
        default=os.path.join(pkg_dir, "config", "nav2_params.yaml"),
    )

    # Per-stage map YAML path
    map_yaml = PythonExpression([
        "'", os.path.join(pkg_dir, "maps"), "/tb3_stage' + '",
        stage, "' + '.yaml'"
    ])

    lifecycle_nodes = [
        "map_server",
        "planner_server",
        "controller_server",
        "behavior_server",
        "bt_navigator",
    ]

    return LaunchDescription([
        DeclareLaunchArgument("stage",        default_value="4",
                              description="Stage number 1-10"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file",  default_value=params_file),

        # ── localization shim ─────────────────────────────────────────────────
        Node(
            package="tb3_nav2",
            executable="pose_publisher_node",
            name="pose_publisher_node",
            output="screen",
            parameters=[{
                "use_sim_time":     use_sim_time,
                "world_name":       "empty",
                "robot_model_name": "waffle_pi",
                "publish_rate_hz":  20.0,
            }],
        ),

        # ── goal forwarder: /goal_pose → Nav2 /navigate_to_pose action ────────
        Node(
            package="tb3_nav2",
            executable="goal_forwarder_node",
            name="goal_forwarder_node",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        # ── Nav2 core servers ─────────────────────────────────────────────────
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[params_file, {
                "use_sim_time":  use_sim_time,
                "yaml_filename": map_yaml,
            }],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[params_file],
        ),

        # ── lifecycle manager ─────────────────────────────────────────────────
        # Configures + activates all the above nodes in the right order.
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_nav2",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart":    True,
                "node_names":   lifecycle_nodes,
            }],
        ),
    ])
