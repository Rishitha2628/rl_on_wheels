#include <algorithm>
#include <cmath>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "tb3_rl_bridge/srv/get_observation.hpp"
#include "tb3_rl_bridge/srv/step.hpp"

using GetObservation = tb3_rl_bridge::srv::GetObservation;
using Step           = tb3_rl_bridge::srv::Step;
using namespace std::chrono_literals;

class EnvBridgeNode : public rclcpp::Node
{
public:
  EnvBridgeNode()
  : Node("env_bridge_node")
  {
    // Parameters
    max_lidar_range_  = declare_parameter("max_lidar_range",  3.5);
    n_lidar_bins_     = declare_parameter("lidar_bins",        36);
    collision_thresh_ = declare_parameter("collision_threshold", 0.2);
    goal_tolerance_   = declare_parameter("goal_tolerance",    0.2);
    step_duration_    = declare_parameter("step_duration",     0.1);
    max_linear_vel_   = declare_parameter("max_linear_vel",    0.26);
    max_angular_vel_  = declare_parameter("max_angular_vel",   1.82);

    lidar_data_.assign(n_lidar_bins_, 1.0f);  // initialise to max-range

    // All callbacks share a reentrant group so the service can sleep
    // while subscriber callbacks keep updating cached sensor data.
    auto cb_group = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto sub_opts = rclcpp::SubscriptionOptions();
    sub_opts.callback_group = cb_group;

    // Subscribers — 2D LiDAR only (no cameras / pointclouds)
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) { on_scan(msg); },
      sub_opts);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { on_odom(msg); },
      sub_opts);

    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { on_goal(msg); },
      sub_opts);

    // Publisher
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    // Services
    get_obs_srv_ = create_service<GetObservation>(
      "/get_observation",
      [this](GetObservation::Request::ConstSharedPtr req,
             GetObservation::Response::SharedPtr    res) {
        (void)req;
        std::lock_guard<std::mutex> lk(mtx_);
        res->observation    = build_observation();
        res->achieved_goal  = {robot_x_, robot_y_};
        res->desired_goal   = {goal_x_,  goal_y_};
        res->success = true;
      },
      rmw_qos_profile_services_default, cb_group);

    step_srv_ = create_service<Step>(
      "/step",
      [this](Step::Request::ConstSharedPtr req, Step::Response::SharedPtr res) {
        handle_step(req, res);
      },
      rmw_qos_profile_services_default, cb_group);

    RCLCPP_INFO(get_logger(), "EnvBridgeNode ready. lidar_bins=%d step_duration=%.2fs",
                n_lidar_bins_, step_duration_);
  }

private:
  // ── sensor callbacks ───────────────────────────────────────────────────────
  void on_scan(sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
  {
    const int   total = static_cast<int>(msg->ranges.size());
    const int   step  = std::max(1, total / n_lidar_bins_);
    const float rmax  = static_cast<float>(max_lidar_range_);

    std::vector<float> bins(n_lidar_bins_);
    for (int i = 0; i < n_lidar_bins_; ++i) {
      int   start   = i * step;
      int   end     = std::min(start + step, total);
      float min_val = rmax;
      for (int j = start; j < end; ++j) {
        float val = msg->ranges[j];
        if (!std::isfinite(val) || val < msg->range_min) val = rmax;
        val = std::clamp(val, 0.0f, rmax);
        min_val = std::min(min_val, val);
      }
      bins[i] = min_val / rmax;       // normalise → [0, 1]
    }

    std::lock_guard<std::mutex> lk(mtx_);
    lidar_data_ = std::move(bins);
    auto min_it    = std::min_element(lidar_data_.begin(), lidar_data_.end());
    min_lidar_     = *min_it * rmax;
    int  min_bin   = static_cast<int>(min_it - lidar_data_.begin());
    // Angle of the closest-obstacle bin in the robot body frame (0 = forward).
    min_lidar_angle_ = (min_bin + 0.5f) * 2.0f * static_cast<float>(M_PI)
                       / static_cast<float>(n_lidar_bins_);
  }

  void on_odom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    robot_x_ = static_cast<float>(msg->pose.pose.position.x);
    robot_y_ = static_cast<float>(msg->pose.pose.position.y);
    lin_vel_ = static_cast<float>(msg->twist.twist.linear.x);
    ang_vel_ = static_cast<float>(msg->twist.twist.angular.z);

    auto& q = msg->pose.pose.orientation;
    float siny = 2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z) +
                         static_cast<float>(q.x) * static_cast<float>(q.y));
    float cosy = 1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y) +
                                 static_cast<float>(q.z) * static_cast<float>(q.z));
    float yaw = std::atan2(siny, cosy);
    cos_yaw_  = std::cos(yaw);
    sin_yaw_  = std::sin(yaw);
  }

  void on_goal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    goal_x_ = static_cast<float>(msg->pose.position.x);
    goal_y_ = static_cast<float>(msg->pose.position.y);
    // New episode: clear any stale done/reward state that arrived between
    // the previous episode ending and this goal being published.
    episode_done_ = false;
    episode_done_info_.clear();
    step_count_ = 0;
    prev_dist_  = std::hypot(robot_x_ - goal_x_, robot_y_ - goal_y_);
    RCLCPP_INFO(get_logger(), "New goal received: odom=(%.2f,%.2f) robot_odom=(%.2f,%.2f)",
                goal_x_, goal_y_, robot_x_, robot_y_);
  }

  // ── observation assembly ───────────────────────────────────────────────────
  // Caller must hold mtx_.
  std::vector<float> build_observation() const
  {
    float dx         = goal_x_ - robot_x_;
    float dy         = goal_y_ - robot_y_;
    float dist       = std::hypot(dx, dy);
    float dist_norm  = std::min(dist, static_cast<float>(max_lidar_range_))
                       / static_cast<float>(max_lidar_range_);
    float goal_angle = std::atan2(dy, dx);
    float yaw        = std::atan2(sin_yaw_, cos_yaw_);
    float cos_goal   = std::cos(goal_angle - yaw);
    float sin_goal   = std::sin(goal_angle - yaw);

    std::vector<float> obs(lidar_data_);  // 36 min lidar bins
    obs.push_back(dist_norm);             // +1
    obs.push_back(cos_goal);              // +1
    obs.push_back(sin_goal);              // +1
    obs.push_back(last_lv_);              // +1 previous action
    obs.push_back(last_av_);              // +1 previous action → total 41
    return obs;
  }

  // ── step service ───────────────────────────────────────────────────────────
  void handle_step(Step::Request::ConstSharedPtr req, Step::Response::SharedPtr res)
  {
    // 1. Clamp and apply action
    float lv = 0.0f, av = 0.0f;
    if (req->action.size() >= 2) {
      lv = std::clamp(req->action[0],
                      0.0f,
                      static_cast<float>(max_linear_vel_));
      av = std::clamp(req->action[1],
                      static_cast<float>(-max_angular_vel_),
                      static_cast<float>( max_angular_vel_));
    }
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x  = lv;
    cmd.angular.z = av;
    cmd_vel_pub_->publish(cmd);

    // 3. Sleep — other executor threads update cached sensor data meanwhile
    std::this_thread::sleep_for(
      std::chrono::milliseconds(static_cast<int>(step_duration_ * 1000.0)));

    // 3. Build response from latest cached data (robot keeps moving at cmd_vel)
    std::lock_guard<std::mutex> lk(mtx_);
    last_lv_ = lv;
    last_av_ = av;

    // Belt-and-suspenders: detect done directly in case subscription message
    // arrived while the mutex was held and got queued past this read.
    ++step_count_;
    float _dx = robot_x_ - goal_x_, _dy = robot_y_ - goal_y_;
    float _dist = std::sqrt(_dx * _dx + _dy * _dy);
    if (step_count_ > 1 && _dist < static_cast<float>(goal_tolerance_)) {
      episode_done_ = true;
      episode_done_info_ = "goal_reached";
    } else if (step_count_ > 1 && min_lidar_ < static_cast<float>(collision_thresh_)) {
      episode_done_ = true;
      episode_done_info_ = "collision";
    }

    // Delta-progress reward: rewards actual movement toward the goal,
    // not instantaneous velocity direction — detours are not penalised.
    float reward;
    if (episode_done_) {
      // Asymmetric: reaching the goal (+100 after Python ×100 scale) is always
      // better than crashing (−50). Symmetric ±100 lets the robot exploit crashes
      // to end unpromising episodes cheaply.
      reward = (episode_done_info_ == "goal_reached") ? 1.0f : -0.5f;
    } else {
      float progress = (prev_dist_ - _dist) / static_cast<float>(max_lidar_range_);
      // Proximity penalty: quadratic, activates only within 0.4 m safety zone.
      // Old threshold was 1.35 m — with obstacles at r~1.4 m the penalty fired on
      // every step, overwhelming progress reward and trapping the robot.
      const float safety_dist = 0.4f;
      float danger = std::max(0.0f, safety_dist - min_lidar_) / safety_dist;
      reward = progress - 0.1f * danger * danger;
    }
    prev_dist_ = _dist;

    res->observation   = build_observation();
    res->achieved_goal = {robot_x_, robot_y_};
    res->desired_goal  = {goal_x_,  goal_y_};
    res->reward        = reward;
    res->done          = episode_done_;
    res->info          = episode_done_info_;

    // Reset done flag so the next step starts clean
    episode_done_ = false;
    episode_done_info_.clear();
  }

  // ── members ────────────────────────────────────────────────────────────────
  std::mutex mtx_;

  // Sensor cache
  std::vector<float> lidar_data_;
  float min_lidar_       = 999.0f;
  float min_lidar_angle_ = 0.0f;   // body-frame angle of closest obstacle bin
  float robot_x_    = 0.0f, robot_y_ = 0.0f;
  float lin_vel_    = 0.0f, ang_vel_ = 0.0f;
  float cos_yaw_    = 1.0f, sin_yaw_ = 0.0f;
  float goal_x_     = 1.0f, goal_y_  = 0.0f;
  float last_lv_    = 0.0f, last_av_ = 0.0f;

  bool  episode_done_     = false;
  std::string episode_done_info_;
  int   step_count_       = 0;
  float prev_dist_        = 0.0f;

  // Parameters
  double max_lidar_range_;
  int    n_lidar_bins_;
  double collision_thresh_;
  double goal_tolerance_;
  double step_duration_;
  double max_linear_vel_;
  double max_angular_vel_;

  // ROS handles
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr    scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr          cmd_vel_pub_;
  rclcpp::Service<GetObservation>::SharedPtr                        get_obs_srv_;
  rclcpp::Service<Step>::SharedPtr                                  step_srv_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EnvBridgeNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
