#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker.hpp"

// Ignition Fortress gz-transport for entity management
#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/boolean.pb.h>
#include <ignition/msgs/entity.pb.h>
#include <ignition/msgs/entity_factory.pb.h>

#include "tb3_rl_bridge/srv/reset_episode.hpp"

using ResetEpisode = tb3_rl_bridge::srv::ResetEpisode;
using namespace std::chrono_literals;

// Box obstacle: 0.15 m deep × 1.2 m wide × 0.6 m tall.
// Placed perpendicular to the robot→goal line so the robot must navigate around it.
static const char * OBS_BOX_SDF = R"(
<?xml version='1.0'?>
<sdf version='1.8'>
  <model name='%NAME%'>
    <static>true</static>
    <link name='link'>
      <pose>0 0 0.3 0 0 0</pose>
      <collision name='col'>
        <geometry><box><size>0.15 1.2 0.6</size></box></geometry>
      </collision>
      <visual name='vis'>
        <geometry><box><size>0.15 1.2 0.6</size></box></geometry>
        <material><ambient>0.4 0.2 0.1 1</ambient><diffuse>0.4 0.2 0.1 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>)";

// Minimal SDF 1.8 sphere — spawned each episode as the goal marker.
static const char * GOAL_SPHERE_SDF = R"(
<?xml version='1.0'?>
<sdf version='1.8'>
  <model name='goal_marker'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><sphere><radius>%RADIUS%</radius></sphere></geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <emissive>0.5 0 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>)";

class ResetNode : public rclcpp::Node
{
public:
  ResetNode()
  : Node("reset_node"), rng_(std::random_device{}()), goal_spawned_(false), obs_spawned_(false)
  {
    tb3_model_     = declare_parameter("tb3_model", std::string("waffle_pi"));
    x_min_         = declare_parameter("world_x_min", -2.0);
    x_max_         = declare_parameter("world_x_max",  2.0);
    y_min_         = declare_parameter("world_y_min", -2.0);
    y_max_         = declare_parameter("world_y_max",  2.0);
    min_dist_      = declare_parameter("robot_min_goal_dist", 0.5);
    sphere_radius_ = declare_parameter("goal_sphere_radius",  0.1);
    world_name_    = declare_parameter("world_name", std::string("empty"));
    n_obstacles_   = declare_parameter("n_obstacles", 0);

    // Reentrant group so the odom callback can fire while the service
    // handler is sleeping after a teleport.
    auto reentrant_group = create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);
    auto sub_opts = rclcpp::SubscriptionOptions();
    sub_opts.callback_group = reentrant_group;

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        odom_x_ = static_cast<float>(msg->pose.pose.position.x);
        odom_y_ = static_cast<float>(msg->pose.pose.position.y);
        auto & q = msg->pose.pose.orientation;
        odom_yaw_ = std::atan2(
          2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z) +
                  static_cast<float>(q.x) * static_cast<float>(q.y)),
          1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y) +
                         static_cast<float>(q.z) * static_cast<float>(q.z)));
      }, sub_opts);

    goal_pub_    = create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);
    marker_pub_  = create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    reset_srv_ = create_service<ResetEpisode>(
      "/reset_episode",
      [this](ResetEpisode::Request::ConstSharedPtr req,
             ResetEpisode::Response::SharedPtr     res) {
        handle_reset(req, res);
      });

    // Delete any leftover goal_marker from a previous run
    delete_entity("goal_marker");

    RCLCPP_INFO(get_logger(), "ResetNode ready (Ignition). model=%s world=%s bounds=[%.1f,%.1f]x[%.1f,%.1f]",
                tb3_model_.c_str(), world_name_.c_str(), x_min_, x_max_, y_min_, y_max_);
  }

private:
  void handle_reset(ResetEpisode::Request::ConstSharedPtr req,
                    ResetEpisode::Response::SharedPtr     res)
  {
    float rx, ry, rtheta, gx, gy;
    std::vector<XY>    obs_pos;
    std::vector<float> obs_yaw;

    if (req->random_pose) {
      sample_all_poses(rx, ry, rtheta, gx, gy, obs_pos, obs_yaw);
    } else {
      rx     = req->spawn_x;
      ry     = req->spawn_y;
      rtheta = req->spawn_theta;
      gx     = req->goal_x;
      gy     = req->goal_y;
    }

    {
      geometry_msgs::msg::Twist zero;
      cmd_vel_pub_->publish(zero);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

    if (!obs_spawned_) {
      spawn_all_obstacles(obs_pos, obs_yaw);
    } else {
      teleport_all_obstacles(obs_pos, obs_yaw);
    }

    bool ok = teleport_robot(rx, ry, rtheta);

    // Wait for physics to settle and for odom to reflect the post-teleport state.
    // With MultiThreadedExecutor the odom callback fires during this sleep, so
    // odom_x_/odom_y_ will hold the fresh value — whether or not the DiffDrive
    // plugin resets its accumulated pose on set_pose (Ignition Fortress behaviour
    // can vary). The robot is stationary (zero cmd_vel above), so there is no
    // movement contaminating the reading.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    float post_odom_x, post_odom_y, post_odom_yaw;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      post_odom_x   = odom_x_;
      post_odom_y   = odom_y_;
      post_odom_yaw = odom_yaw_;
    }

    // Transform goal from world frame to odom frame, accounting for both the
    // translational offset and the rotational offset between frames.
    // The DiffDrive plugin does not reset its accumulated heading when set_pose
    // is called, so the odom frame can be rotated by dtheta = post_odom_yaw - rtheta
    // relative to the world frame. Ignoring this rotation causes goal_odom to point
    // in the wrong direction, which is why the robot appears to "reach" the wrong spot.
    float dtheta    = post_odom_yaw - rtheta;
    float cos_d     = std::cos(dtheta);
    float sin_d     = std::sin(dtheta);
    float dgx       = gx - rx;
    float dgy       = gy - ry;
    float goal_odom_x = post_odom_x + cos_d * dgx - sin_d * dgy;
    float goal_odom_y = post_odom_y + sin_d * dgx + cos_d * dgy;
    publish_goal(goal_odom_x, goal_odom_y);
    std::thread([this, gx, gy]() { update_goal_marker(gx, gy); }).detach();

    res->success = ok;
    res->info    = ok ? "reset_ok" : "teleport_failed";

    RCLCPP_INFO(get_logger(),
                "Reset: robot_world=(%.2f,%.2f,θ=%.2f) goal_world=(%.2f,%.2f) | "
                "post_odom=(%.2f,%.2f,θ=%.2f) dtheta=%.2f goal_odom=(%.2f,%.2f)",
                rx, ry, rtheta, gx, gy,
                post_odom_x, post_odom_y, post_odom_yaw, dtheta,
                goal_odom_x, goal_odom_y);
  }

  // ── pose sampling ──────────────────────────────────────────────────────────
  using XY = std::pair<float, float>;
  static std::string obs_name(int i) { return "dyn_obs_" + std::to_string(i); }

  // Robot spawns on one side, goal on the other, box obstacle between them.
  void sample_all_poses(float & rx, float & ry, float & rtheta,
                        float & gx, float & gy,
                        std::vector<XY> & obs_pos, std::vector<float> & obs_yaw)
  {
    std::uniform_real_distribution<float> dxy(x_min_, x_max_);
    std::uniform_real_distribution<float> ddir(-M_PI, M_PI);
    std::uniform_real_distribution<float> d1_dist(0.7f, 1.1f);
    std::uniform_real_distribution<float> d2_dist(0.7f, 1.1f);

    int tries = 0;
    do {
      rx         = dxy(rng_); ry = dxy(rng_);
      float dir  = ddir(rng_);
      float d1   = d1_dist(rng_);   // robot → obstacle
      float d2   = d2_dist(rng_);   // obstacle → goal
      float ox   = rx + d1 * std::cos(dir);
      float oy   = ry + d1 * std::sin(dir);
      gx         = std::clamp(ox + d2 * std::cos(dir),
                              static_cast<float>(x_min_), static_cast<float>(x_max_));
      gy         = std::clamp(oy + d2 * std::sin(dir),
                              static_cast<float>(y_min_), static_cast<float>(y_max_));
      obs_pos    = {{ox, oy}};
      obs_yaw    = {dir + static_cast<float>(M_PI_2)};
    } while (++tries < 200 &&
             std::hypot(gx - rx, gy - ry) < static_cast<float>(min_dist_));

    rtheta = std::atan2(gy - ry, gx - rx);

    if (n_obstacles_ == 0) { obs_pos.clear(); obs_yaw.clear(); }
  }

  void spawn_all_obstacles(const std::vector<XY> & poses,
                           const std::vector<float> & yaws)
  {
    for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
      std::string sdf(OBS_BOX_SDF);
      auto p = sdf.find("%NAME%");
      if (p != std::string::npos) sdf.replace(p, 6, obs_name(i));
      spawn_entity(sdf, poses[i].first, poses[i].second, 0.0f, yaws[i]);
    }
    obs_spawned_ = true;
  }

  void teleport_all_obstacles(const std::vector<XY> & poses,
                              const std::vector<float> & yaws)
  {
    for (int i = 0; i < static_cast<int>(poses.size()); ++i)
      teleport_entity(obs_name(i), poses[i].first, poses[i].second, yaws[i], 0.0f);
  }

  // ── Ignition gz-transport: teleport any entity ────────────────────────────
  bool teleport_entity(const std::string & name, float x, float y,
                       float theta = 0.0f, float z = 0.0f)
  {
    ignition::msgs::Pose req;
    req.set_name(name);
    req.mutable_position()->set_x(x);
    req.mutable_position()->set_y(y);
    req.mutable_position()->set_z(z);
    req.mutable_orientation()->set_x(0.0);
    req.mutable_orientation()->set_y(0.0);
    req.mutable_orientation()->set_z(std::sin(theta * 0.5));
    req.mutable_orientation()->set_w(std::cos(theta * 0.5));
    ignition::msgs::Boolean res; bool result = false;
    ign_node_.Request("/world/" + world_name_ + "/set_pose", req, 2000, res, result);
    return result;
  }

  // ── Ignition gz-transport: teleport robot ──────────────────────────────────
  bool teleport_robot(float x, float y, float theta)
  {
    ignition::msgs::Pose req;
    req.set_name(tb3_model_);
    req.mutable_position()->set_x(x);
    req.mutable_position()->set_y(y);
    req.mutable_position()->set_z(0.0);
    req.mutable_orientation()->set_x(0.0);
    req.mutable_orientation()->set_y(0.0);
    req.mutable_orientation()->set_z(std::sin(theta * 0.5));
    req.mutable_orientation()->set_w(std::cos(theta * 0.5));

    ignition::msgs::Boolean res;
    bool result = false;
    std::string service = "/world/" + world_name_ + "/set_pose";

    bool executed = ign_node_.Request(service, req, 2000, res, result);
    if (!executed || !result) {
      RCLCPP_WARN(get_logger(), "set_pose failed for %s (executed=%d result=%d)",
                  tb3_model_.c_str(), executed, result);
      return false;
    }
    return true;
  }

  // ── Ignition gz-transport: spawn entity ───────────────────────────────────
  bool spawn_entity(const std::string & sdf_str, float x, float y, float z,
                    float yaw = 0.0f)
  {
    ignition::msgs::EntityFactory req;
    req.set_sdf(sdf_str);
    req.mutable_pose()->mutable_position()->set_x(x);
    req.mutable_pose()->mutable_position()->set_y(y);
    req.mutable_pose()->mutable_position()->set_z(z);
    req.mutable_pose()->mutable_orientation()->set_z(std::sin(yaw * 0.5f));
    req.mutable_pose()->mutable_orientation()->set_w(std::cos(yaw * 0.5f));

    std::string service = "/world/" + world_name_ + "/create";

    for (int attempt = 1; attempt <= 3; ++attempt) {
      ignition::msgs::Boolean res;
      bool result = false;
      bool executed = ign_node_.Request(service, req, 3000, res, result);
      if (executed && result) return true;
      RCLCPP_WARN(get_logger(), "spawn_entity attempt %d/3 failed (executed=%d result=%d)",
                  attempt, executed, result);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
  }

  // ── Ignition gz-transport: delete entity ──────────────────────────────────
  void delete_entity(const std::string & name)
  {
    ignition::msgs::Entity req;
    req.set_name(name);
    req.set_type(ignition::msgs::Entity::MODEL);

    ignition::msgs::Boolean res;
    bool result = false;
    std::string service = "/world/" + world_name_ + "/remove";

    ign_node_.Request(service, req, 2000, res, result);
  }

  // ── goal marker management ─────────────────────────────────────────────────
  void update_goal_marker(float gx, float gy)
  {
    if (goal_spawned_) {
      delete_entity("goal_marker");
      std::this_thread::sleep_for(200ms);
    }

    std::string sdf(GOAL_SPHERE_SDF);
    auto pos = sdf.find("%RADIUS%");
    if (pos != std::string::npos)
      sdf.replace(pos, 8, std::to_string(sphere_radius_));

    if (spawn_entity(sdf, gx, gy, static_cast<float>(sphere_radius_)))
      goal_spawned_ = true;

    // RViz marker
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "map";
    m.header.stamp    = now();
    m.ns = "goal"; m.id = 0;
    m.type   = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = gx;
    m.pose.position.y = gy;
    m.pose.position.z = sphere_radius_;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = sphere_radius_ * 2.0;
    m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
    marker_pub_->publish(m);
  }

  void publish_goal(float gx, float gy)
  {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = "odom";
    ps.header.stamp    = now();
    ps.pose.position.x = gx;
    ps.pose.position.y = gy;
    ps.pose.orientation.w = 1.0;
    goal_pub_->publish(ps);
  }

  // ── members ────────────────────────────────────────────────────────────────
  std::mutex mtx_;
  std::string tb3_model_, world_name_;
  double x_min_, x_max_, y_min_, y_max_;
  double min_dist_, sphere_radius_;
  float odom_x_ = 0.0f, odom_y_ = 0.0f, odom_yaw_ = 0.0f;

  int  n_obstacles_;
  std::mt19937 rng_;
  bool goal_spawned_;
  bool obs_spawned_;
  std::vector<std::pair<float,float>> obs_xy_;

  ignition::transport::Node ign_node_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr   goal_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr   marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr         cmd_vel_pub_;
  rclcpp::Service<ResetEpisode>::SharedPtr                        reset_srv_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ResetNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
