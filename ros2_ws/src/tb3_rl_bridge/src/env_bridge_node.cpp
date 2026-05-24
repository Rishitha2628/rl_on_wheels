#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose_v.pb.h>

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
    max_lidar_range_   = declare_parameter("max_lidar_range",  3.5);
    n_lidar_bins_      = declare_parameter("lidar_bins",        36);
    collision_thresh_  = declare_parameter("collision_threshold", 0.2);
    goal_tolerance_    = declare_parameter("goal_tolerance",    0.2);
    step_duration_     = declare_parameter("step_duration",     0.1);
    min_linear_vel_    = declare_parameter("min_linear_vel",   0.0);
    max_linear_vel_    = declare_parameter("max_linear_vel",    0.26);
    max_angular_vel_   = declare_parameter("max_angular_vel",   1.82);
    clearance_threshold_ = declare_parameter("clearance_threshold", 0.5);

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

    // Moving-obstacle world poses (published by dynamic_obstacle_node).
    // Lets us compute geometric robot-to-obstacle distance for r_obstacle,
    // exactly like drlnav's /obstacle/odom_obs feed.
    obstacles_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "/obstacle_poses", 10,
      [this](geometry_msgs::msg::PoseArray::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        obstacle_positions_.clear();
        obstacle_positions_.reserve(msg->poses.size());
        for (const auto & p : msg->poses) {
          obstacle_positions_.emplace_back(
            static_cast<float>(p.position.x),
            static_cast<float>(p.position.y));
        }
      },
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
        res->success  = true;
        res->goal_seq = goal_seq_;     // sequence counter for Python's wait_new_goal()
      },
      rmw_qos_profile_services_default, cb_group);

    step_srv_ = create_service<Step>(
      "/step",
      [this](Step::Request::ConstSharedPtr req, Step::Response::SharedPtr res) {
        handle_step(req, res);
      },
      rmw_qos_profile_services_default, cb_group);

    // Subscribe to Ignition's scene-broadcaster pose feed for the robot's
    // TRUE world pose (matches drlnav's effective odom — no DiffDrive drift).
    std::string world_name = declare_parameter("world_name", std::string("empty"));
    robot_model_name_      = declare_parameter("robot_model_name",
                                               std::string("waffle_pi"));
    std::string pose_topic = "/world/" + world_name + "/dynamic_pose/info";
    bool ign_ok = ign_node_.Subscribe(pose_topic, &EnvBridgeNode::on_ign_pose, this);
    RCLCPP_INFO(get_logger(),
                "EnvBridgeNode ready. lidar_bins=%d step_duration=%.2fs "
                "ign_pose_topic=%s subscribed=%d robot=%s",
                n_lidar_bins_, step_duration_,
                pose_topic.c_str(), ign_ok ? 1 : 0,
                robot_model_name_.c_str());
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
        // NaN/inf = sensor confused → assume far (no reading).
        // Below range_min = obstacle is TOO CLOSE for the sensor to resolve
        // → treat as range_min so collision detection still fires.
        if (!std::isfinite(val)) {
          val = rmax;
        } else if (val < msg->range_min) {
          val = msg->range_min;
        }
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
    // DiffDrive's /odom drifts (wheel-encoder based). We use it ONLY for
    // velocity readings — position + orientation come from Ignition's
    // scene-broadcaster pose feed (see on_ign_pose), which is ground truth.
    std::lock_guard<std::mutex> lk(mtx_);
    lin_vel_ = static_cast<float>(msg->twist.twist.linear.x);
    ang_vel_ = static_cast<float>(msg->twist.twist.angular.z);
  }

  // Ignition transport callback — receives ALL entity poses in WORLD frame
  // from the scene broadcaster. We filter for the robot.
  void on_ign_pose(const ignition::msgs::Pose_V & msg)
  {
    for (int i = 0; i < msg.pose_size(); ++i) {
      const auto & pose = msg.pose(i);
      if (pose.name() != robot_model_name_) continue;
      std::lock_guard<std::mutex> lk(mtx_);
      robot_x_ = static_cast<float>(pose.position().x());
      robot_y_ = static_cast<float>(pose.position().y());
      const auto & q = pose.orientation();
      float siny = 2.0f * (static_cast<float>(q.w()) * static_cast<float>(q.z()) +
                           static_cast<float>(q.x()) * static_cast<float>(q.y()));
      float cosy = 1.0f - 2.0f * (static_cast<float>(q.y()) * static_cast<float>(q.y()) +
                                   static_cast<float>(q.z()) * static_cast<float>(q.z()));
      float yaw = std::atan2(siny, cosy);
      cos_yaw_ = std::cos(yaw);
      sin_yaw_ = std::sin(yaw);
      return;
    }
  }

  void on_goal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    goal_x_ = static_cast<float>(msg->pose.position.x);
    goal_y_ = static_cast<float>(msg->pose.position.y);
    new_goal_ = true;     // drlnav goal_pose_callback equivalent
    ++goal_seq_;          // monotonic counter — Python's wait_new_goal polls this
    // New episode: clear any stale done/reward state that arrived between
    // the previous episode ending and this goal being published.
    episode_done_ = false;
    episode_done_info_.clear();
    step_count_    = 0;
    prev_dist_     = std::hypot(robot_x_ - goal_x_, robot_y_ - goal_y_);
    // Initial goal distance — anchor for asymmetric r_distance shaping.
    // Clamp away from 0 to keep the 2*d0/(d0+d) denominator well-behaved.
    goal_dist_initial_ = std::max(prev_dist_, 0.05f);
    prev_min_lidar_ = min_lidar_;
    RCLCPP_INFO(get_logger(), "New goal received: odom=(%.2f,%.2f) robot_odom=(%.2f,%.2f) d0=%.2f",
                goal_x_, goal_y_, robot_x_, robot_y_, goal_dist_initial_);
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
    float goal_body  = goal_angle - yaw;   // body-frame angle to goal
    float cos_goal   = std::cos(goal_body);
    float sin_goal   = std::sin(goal_body);

    // Min lidar reading in a ±20° cone around the goal direction (in body frame).
    // Gives the policy an explicit "is the path to the goal blocked?" feature
    // instead of forcing it to infer this from the joint of lidar + goal angle.
    float goal_body_pos = std::fmod(
      goal_body + 2.0f * static_cast<float>(M_PI),
      2.0f * static_cast<float>(M_PI));
    int center_bin = static_cast<int>(
      goal_body_pos / (2.0f * static_cast<float>(M_PI)) * n_lidar_bins_)
      % n_lidar_bins_;
    const int goal_half_window = 2;  // ±2 bins = ±20° at 36 bins → 5 bins
    float goal_path_min = 1.0f;
    for (int b = -goal_half_window; b <= goal_half_window; ++b) {
      int bin = (center_bin + b + n_lidar_bins_) % n_lidar_bins_;
      goal_path_min = std::min(goal_path_min, lidar_data_[bin]);
    }

    std::vector<float> obs(lidar_data_);  // 36 min lidar bins
    obs.push_back(dist_norm);             // +1
    obs.push_back(cos_goal);              // +1
    obs.push_back(sin_goal);              // +1
    obs.push_back(goal_path_min);         // +1 obstacle-on-path-to-goal signal
    obs.push_back(last_lv_);              // +1 previous action
    obs.push_back(last_av_);              // +1 previous action → total 42
    return obs;
  }

  // ── step service ───────────────────────────────────────────────────────────
  void handle_step(Step::Request::ConstSharedPtr req, Step::Response::SharedPtr res)
  {
    // 1. Clamp and apply action
    float lv = 0.0f, av = 0.0f;
    if (req->action.size() >= 2) {
      lv = std::clamp(req->action[0],
                      static_cast<float>(min_linear_vel_),
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
    // drlnav grace period — first GRACE_STEPS post-reset don't trigger termination,
    // so phantom collisions from teleport settling can't end the episode prematurely.
    const int GRACE_STEPS = 30;
    if (step_count_ > GRACE_STEPS && _dist < static_cast<float>(goal_tolerance_)) {
      episode_done_ = true;
      episode_done_info_ = "goal_reached";
      RCLCPP_INFO(get_logger(),
                  "[done] goal_reached step=%d dist=%.3f tol=%.2f "
                  "robot_odom=(%.2f,%.2f) goal_odom=(%.2f,%.2f)",
                  step_count_, _dist, goal_tolerance_,
                  robot_x_, robot_y_, goal_x_, goal_y_);
    } else if (step_count_ > GRACE_STEPS && min_lidar_ < static_cast<float>(collision_thresh_)) {
      episode_done_ = true;
      episode_done_info_ = "collision";
      RCLCPP_INFO(get_logger(),
                  "[done] collision step=%d min_lidar=%.3f thresh=%.2f "
                  "robot_odom=(%.2f,%.2f)",
                  step_count_, min_lidar_, collision_thresh_,
                  robot_x_, robot_y_);
    }

    // Front cone min lidar (±~30°) for progress-gating.
    // Bin layout: bin 0 covers 0°-9°, bin 35 covers 350°-359°.
    // half_window=3 → bins {33,34,35,0,1,2,3} = 7 bins, ~70° cone.
    const float clearance_threshold = static_cast<float>(clearance_threshold_);
    const int   half_window = 3;
    float front_min_m = static_cast<float>(max_lidar_range_);
    for (int b = -half_window; b <= half_window; ++b) {
      int bin = (b + n_lidar_bins_) % n_lidar_bins_;
      front_min_m = std::min(front_min_m,
        lidar_data_[bin] * static_cast<float>(max_lidar_range_));
    }
    bool  progress_gated_flag = (front_min_m <= clearance_threshold);

    // drlnav-style stop-on-episode-end: when the robot reaches a goal or
    // collides, publish a zero cmd_vel BEFORE returning. Otherwise DiffDrive
    // keeps applying the previous action through the wait_new_goal /
    // time.sleep(0.5) window in the Python env — robot drifts past the goal
    // and into walls during the goal-switch.
    if (episode_done_) {
      geometry_msgs::msg::Twist stop;
      cmd_vel_pub_->publish(stop);
    }

    // ── reward (drlnav get_reward_A port) ──────────────────────────────────
    // Body-frame goal angle, wrapped to [-pi, pi] for r_yaw.
    float dx_g = goal_x_ - robot_x_;
    float dy_g = goal_y_ - robot_y_;
    float yaw  = std::atan2(sin_yaw_, cos_yaw_);
    float goal_body = std::atan2(dy_g, dx_g) - yaw;
    while (goal_body >  static_cast<float>(M_PI)) goal_body -= 2.0f * static_cast<float>(M_PI);
    while (goal_body < -static_cast<float>(M_PI)) goal_body += 2.0f * static_cast<float>(M_PI);

    float reward;
    float progress_applied = 0.0f;
    if (episode_done_) {
      reward = (episode_done_info_ == "goal_reached") ? 2500.0f : -2000.0f;
    } else {
      // r_yaw: face the goal. Range [-pi, 0].
      float r_yaw = -std::abs(goal_body);

      // r_vangular: penalise spinning. Range [-max_av^2, 0].
      float r_vangular = -(av * av);

      // r_vlinear: pull policy toward max forward speed.
      // -((max_lv - lv) * 10)^2 — at lv=max_lv → 0, at lv=0 → -((max_lv*10)^2).
      float lv_diff   = static_cast<float>(max_linear_vel_) - lv;
      float r_vlinear = -(lv_diff * 10.0f) * (lv_diff * 10.0f);

      // r_distance: asymmetric shaping. Range [-1, 1].
      // 0 at start, →+1 at goal, →-1 far away. Grows nonlinearly near goal.
      float d0         = std::max(goal_dist_initial_, 0.05f);
      float r_distance = 2.0f * d0 / (d0 + _dist) - 1.0f;

      // r_obstacle (drlnav exact): geometric distance from robot to nearest
      // MOVING obstacle. Fires at < 0.22 m. Wall proximity is intentionally
      // excluded — only moving cylinders trigger this penalty, matching
      // drlnav's obstacle_distance check.
      // When obstacle_positions_ is empty (stages 1, 2 — no moving obstacles)
      // this defaults to 0 (no penalty).
      float min_obs_dist = std::numeric_limits<float>::infinity();
      for (const auto & op : obstacle_positions_) {
        float odx = op.first  - robot_x_;
        float ody = op.second - robot_y_;
        float od  = std::sqrt(odx * odx + ody * ody);
        if (od < min_obs_dist) min_obs_dist = od;
      }
      float r_obstacle = (min_obs_dist < 0.22f) ? -20.0f : 0.0f;

      // Constant per-step penalty (drlnav: -1). Encourages task completion.
      reward = r_yaw + r_vangular + r_vlinear + r_distance + r_obstacle - 1.0f;

      // Telemetry: report the shaping term most directly tied to progress.
      progress_applied = r_distance;
    }
    // front_min_m / progress_gated_flag are no longer used by the reward
    // itself; kept in the response for info / logging only.
    (void)progress_gated_flag;
    prev_dist_     = _dist;
    prev_min_lidar_ = min_lidar_;

    res->observation     = build_observation();
    res->achieved_goal   = {robot_x_, robot_y_};
    res->desired_goal    = {goal_x_,  goal_y_};
    res->reward          = reward;
    res->done            = episode_done_;
    res->info            = episode_done_info_;
    res->progress_reward = progress_applied;
    res->front_min       = front_min_m;
    res->progress_gated  = progress_gated_flag;

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
  float prev_min_lidar_   = 999.0f;
  float goal_dist_initial_ = 1.0f;  // d0 for r_distance shaping; set in on_goal()
  bool  new_goal_         = false;  // drlnav: flipped to true on /goal_pose receipt
  uint32_t goal_seq_      = 0;      // monotonic counter, increments on each on_goal

  // Parameters
  double max_lidar_range_;
  int    n_lidar_bins_;
  double collision_thresh_;
  double goal_tolerance_;
  double step_duration_;
  double min_linear_vel_;
  double max_linear_vel_;
  double max_angular_vel_;
  double clearance_threshold_;

  // Moving-obstacle world positions (from dynamic_obstacle_node), used by
  // r_obstacle to fire the drlnav-style geometric proximity penalty.
  std::vector<std::pair<float, float>> obstacle_positions_;

  // Ignition pose subscriber — truth source for robot world pose
  // (replaces drifted DiffDrive /odom for position tracking).
  ignition::transport::Node ign_node_;
  std::string               robot_model_name_;

  // ROS handles
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr    scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr   obstacles_sub_;
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
