import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        Node(
            package="tb3_rl_bridge",
            executable="env_bridge_node",
            name="env_bridge_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "max_lidar_range": 3.5,
                "lidar_bins": 36,
                "collision_threshold": 0.0,
                "step_duration": 0.05,
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
                "collision_threshold": 0.0,
                "max_lidar_range": 3.5,
                "goal_tolerance": 0.2,
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
                "world_x_max": 2.0,
                "world_y_min": -2.0,
                "world_y_max": 2.0,
                "robot_min_goal_dist": 0.5,
                "goal_sphere_radius": 0.1,
            }],
        ),
    ])
