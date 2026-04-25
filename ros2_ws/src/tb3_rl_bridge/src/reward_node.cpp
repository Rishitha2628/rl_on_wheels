#include <cmath>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float32.hpp"

class RewardNode : public rclcpp::Node
{
public:
  RewardNode()
  : Node("reward_node"), episode_done_(false)
  {
    // Parameters (must match bridge.launch.py)
    collision_threshold_    = declare_parameter("collision_threshold",    0.2);
    max_lidar_range_        = declare_parameter("max_lidar_range",        3.5);
    goal_tolerance_         = declare_parameter("goal_tolerance",         0.2);
    collision_penalty_      = declare_parameter("collision_penalty",    -1.0);
    goal_reward_            = declare_parameter("goal_reward",            1.0);
    time_penalty_           = declare_parameter("time_penalty",          -0.005);
    distance_reward_scale_  = declare_parameter("distance_reward_scale",  1.0);

    // Subscribers — LiDAR only for collision; odom for position; goal for target
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) { on_scan(msg); });

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { on_odom(msg); });

    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { on_goal(msg); });

    // Publishers
    reward_pub_ = create_publisher<std_msgs::msg::Float32>("/reward", 10);

    RCLCPP_INFO(get_logger(),
                "RewardNode ready. collision_threshold=%.2f goal_tolerance=%.2f",
                collision_threshold_, goal_tolerance_);
  }

private:
  void on_scan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
  {
    float min_range = static_cast<float>(max_lidar_range_);
    for (float r : msg->ranges) {
      if (std::isfinite(r) && r < min_range) min_range = r;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    min_lidar_range_ = min_range;
    compute_and_publish();
  }

  void on_odom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    robot_x_ = static_cast<float>(msg->pose.pose.position.x);
    robot_y_ = static_cast<float>(msg->pose.pose.position.y);
    compute_and_publish();
  }

  void on_goal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    goal_x_ = static_cast<float>(msg->pose.position.x);
    goal_y_ = static_cast<float>(msg->pose.position.y);
    episode_done_ = false;
  }

  // Caller must hold mtx_.
  void compute_and_publish()
  {
    if (episode_done_) return;

    float dx   = robot_x_ - goal_x_;
    float dy   = robot_y_ - goal_y_;
    float dist = std::sqrt(dx * dx + dy * dy);

    // Potential-based reward — stateless, so HER can relabel goals correctly.
    float dist_norm = std::min(dist, static_cast<float>(max_lidar_range_));
    float reward = static_cast<float>(
      -dist_norm / max_lidar_range_ * distance_reward_scale_ + time_penalty_);

    // Goal reached — override reward with success bonus
    if (dist < static_cast<float>(goal_tolerance_)) {
      reward = static_cast<float>(goal_reward_);
      episode_done_ = true;
    }

    // Collision — override reward with collision penalty
    if (min_lidar_range_ < static_cast<float>(collision_threshold_)) {
      reward = static_cast<float>(collision_penalty_);
      episode_done_ = true;
    }

    std_msgs::msg::Float32 r_msg;
    r_msg.data = reward;
    reward_pub_->publish(r_msg);
  }

  // ── members ────────────────────────────────────────────────────────────────
  std::mutex mtx_;

  float robot_x_        = 0.0f, robot_y_ = 0.0f;
  float goal_x_         = 1.0f, goal_y_  = 0.0f;
  float min_lidar_range_ = 999.0f;
  bool  episode_done_;

  double collision_threshold_;
  double max_lidar_range_;
  double goal_tolerance_;
  double collision_penalty_;
  double goal_reward_;
  double time_penalty_;
  double distance_reward_scale_;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr    scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr             reward_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RewardNode>());
  rclcpp::shutdown();
  return 0;
}
