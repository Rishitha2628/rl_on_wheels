// dynamic_obstacle_node — drives keyframed PoseAnimation paths for moving
// obstacles via Ignition Fortress /world/<name>/set_pose. Each obstacle
// has a fixed base position (set in the stage SDF); the keyframes are
// relative translations from that base, looped indefinitely.
//
// The launch file selects which obstacles to animate (1-based indices).
// Stage 1-3   → none
// Stage 4,7-10→ {1, 2}
// Stage 5, 6  → {1, 2, 3, 4, 5, 6}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_array.hpp"

#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/boolean.pb.h>

using namespace std::chrono_literals;

struct Keyframe { double t; double dx; double dy; };

struct ObstacleSpec {
  int idx;                           // 1..6 — name suffix
  double base_x, base_y;             // initial pose set in the stage SDF
  std::vector<Keyframe> keys;        // sorted by t; last == cycle end
};

// Obstacle keyframe tables (translations relative to base pose).
static const std::vector<Keyframe> KEYS_1 = {
  {  0,  0.0,  0.0}, { 10, -0.5, -1.0}, { 50, -3.5, -1.0},
  { 70, -3.7, -3.0}, { 90, -3.5, -1.0}, {130, -0.5, -1.0},
  {140,  0.0,  0.0}, {160,  0.0,  0.0},
};
static const std::vector<Keyframe> KEYS_2 = {
  {  0, 0.0, 0.0}, { 10, 0.7, 0.2}, { 40, 2.5, 3.5},
  { 55, 0.3, 3.5}, { 85, 3.5, 1.8}, {100, 3.5, 0.0},
  {110, 2.0, 0.5}, {115, 1.5, 1.0}, {120, 1.0, 0.5},
  {125, 0.5, 0.1}, {130, 0.0, 0.0}, {140, 0.0, 0.0},
};
static const std::vector<Keyframe> KEYS_3 = {
  {  0,  0.0, 0.0}, { 10, -1.0, 0.2}, { 40, -2.0, 1.0},
  { 55, -3.5, 0.0}, { 85, -2.5, 1.5}, {110,  0.0, 0.0},
  {130, -1.0, 2.0}, {145, -2.0, 1.0}, {165,  0.0, 0.0},
};
static const std::vector<Keyframe> KEYS_4 = {
  {  0, 0.0, 0.0}, { 10, 0.0, -3.2}, { 30, 2.0, -2.7},
  { 40, 0.0,  0.0}, { 60, 0.0, -3.2}, { 80, 2.0, -2.7},
  {110, 0.0,  0.0}, {130, 0.0, -3.2}, {150, 2.0, -2.7},
  {170, 0.0,  0.0},
};
static const std::vector<Keyframe> KEYS_5 = {
  {  0, 0.0, 0.0}, { 10, 0.7,  1.0}, { 40, 2.5,  2.0},
  { 55, 0.0, 2.0}, { 85, 0.0, -1.0}, {110, 2.0,  0.0},
  {125, 4.0, 0.0}, {145, 3.0, -2.0}, {170, 2.0,  0.0},
  {185, 2.0, 2.0}, {205, 0.0,  0.0},
};
static const std::vector<Keyframe> KEYS_6 = {
  {  0,  0.0, 0.0}, { 10, -1.0,  0.0}, { 40, -1.0,  2.0},
  { 55, -1.5, 0.0}, { 85,  0.0,  2.0}, {120, -4.0, -1.8},
  {130, -3.0, -1.8},{145, -2.5,  1.0}, {170,  0.0,  0.0},
};

// Stage-3 small-amplitude oscillations. Each obstacle moves within a small
// radius of its base via 4 different phase-offset oscillations
// (±0.4 m, periods 12-18 s) so the cylinders don't move in lockstep.
static const std::vector<Keyframe> KEYS_S3_1 = {  // N-S, 16s
  {  0, 0.0,  0.0}, {  4, 0.0,  0.4}, {  8, 0.0, 0.0},
  { 12, 0.0, -0.4}, { 16, 0.0,  0.0},
};
static const std::vector<Keyframe> KEYS_S3_2 = {  // E-W, 14s
  {  0,  0.0, 0.0}, {  3.5,  0.4, 0.0}, {  7,  0.0, 0.0},
  { 10.5, -0.4, 0.0}, { 14,  0.0, 0.0},
};
static const std::vector<Keyframe> KEYS_S3_3 = {  // diagonal, 18s
  {  0, 0.0, 0.0}, {  4.5,  0.3,  0.3}, {  9,  0.0, 0.0},
  { 13.5, -0.3, -0.3}, { 18, 0.0, 0.0},
};
static const std::vector<Keyframe> KEYS_S3_4 = {  // anti-diagonal, 12s
  {  0, 0.0, 0.0}, {  3,  0.3, -0.3}, {  6, 0.0, 0.0},
  {  9, -0.3,  0.3}, { 12, 0.0, 0.0},
};

static const std::vector<Keyframe> & keys_for_keyframe(int idx)
{
  switch (idx) {
    case 1: return KEYS_1;
    case 2: return KEYS_2;
    case 3: return KEYS_3;
    case 4: return KEYS_4;
    case 5: return KEYS_5;
    case 6: return KEYS_6;
    default: return KEYS_1;
  }
}

static const std::vector<Keyframe> & keys_for_stage3(int idx)
{
  switch (idx) {
    case 1: return KEYS_S3_1;
    case 2: return KEYS_S3_2;
    case 3: return KEYS_S3_3;
    case 4: return KEYS_S3_4;
    default: return KEYS_S3_1;
  }
}

class DynamicObstacleNode : public rclcpp::Node
{
public:
  DynamicObstacleNode()
  : Node("dynamic_obstacle_node")
  {
    world_name_  = declare_parameter("world_name",     std::string("empty"));
    update_rate_ = declare_parameter("update_rate_hz", 20.0);
    z_height_    = declare_parameter("obs_z",          0.25);
    // Slow-motion factor for the keyframe playback. time_scale=1.0 plays at
    // the authored speed; >1 stretches the timeline so cylinders move slower.
    time_scale_  = declare_parameter("time_scale",     1.0);
    // "keyframe" (default) → KEYS_1..KEYS_6 large-path animations
    // "stage3"             → small-amplitude oscillations for the 4 cylinders
    //                        at (±1, ±1)
    motion_mode_ = declare_parameter("motion_mode", std::string("keyframe"));

    // active_obstacles: comma-separated indices like "1,2" or "1,2,3,4,5,6".
    std::string active_str = declare_parameter("active_obstacles", std::string(""));
    // Per-obstacle base positions (where the stage SDF placed each model).
    // 6 doubles for the x's, then 6 for the y's, default matches stage 5/6.
    auto base_x = declare_parameter("obstacle_base_x",
        std::vector<double>{2.0, -2.0,  2.0, -2.0, -2.0, 2.0});
    auto base_y = declare_parameter("obstacle_base_y",
        std::vector<double>{2.0, -2.0, -2.0,  2.0,  0.0, 0.0});

    parse_active(active_str, base_x, base_y);

    // Publish each moving obstacle's current world pose so env_bridge_node
    // can compute geometric robot-to-obstacle distance for the r_obstacle
    // reward term.
    obstacle_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      "/obstacle_poses", 10);

    start_time_ = this->now();
    auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / std::max(1.0, update_rate_)));
    timer_ = create_wall_timer(period, [this]() { tick(); });

    std::ostringstream oss;
    for (auto & o : obstacles_) {
      oss << " #" << o.idx << "@(" << o.base_x << "," << o.base_y << ")";
    }
    RCLCPP_INFO(get_logger(),
                "DynamicObstacleNode ready. world=%s rate=%.1fHz active=[%s ]",
                world_name_.c_str(), update_rate_, oss.str().c_str());
  }

private:
  void parse_active(const std::string & csv,
                    const std::vector<double> & base_x,
                    const std::vector<double> & base_y)
  {
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      try {
        int idx = std::stoi(tok);
        if (idx < 1 || idx > 6) continue;
        ObstacleSpec spec;
        spec.idx = idx;
        spec.base_x = (idx - 1 < (int)base_x.size()) ? base_x[idx - 1] : 0.0;
        spec.base_y = (idx - 1 < (int)base_y.size()) ? base_y[idx - 1] : 0.0;
        spec.keys   = (motion_mode_ == "stage3")
                      ? keys_for_stage3(idx)
                      : keys_for_keyframe(idx);
        obstacles_.push_back(spec);
      } catch (...) { /* skip bad tokens */ }
    }
  }

  static std::pair<double, double> interp(const std::vector<Keyframe> & keys,
                                          double t)
  {
    if (keys.empty()) return {0.0, 0.0};
    double cycle = keys.back().t;
    if (cycle <= 0.0) return {keys.front().dx, keys.front().dy};
    double s = std::fmod(t, cycle);
    // Find the segment [keys[i], keys[i+1]] containing s.
    for (size_t i = 0; i + 1 < keys.size(); ++i) {
      if (s >= keys[i].t && s <= keys[i+1].t) {
        double dt = keys[i+1].t - keys[i].t;
        double a  = (dt > 0.0) ? (s - keys[i].t) / dt : 0.0;
        double dx = keys[i].dx + a * (keys[i+1].dx - keys[i].dx);
        double dy = keys[i].dy + a * (keys[i+1].dy - keys[i].dy);
        return {dx, dy};
      }
    }
    return {keys.back().dx, keys.back().dy};
  }

  void tick()
  {
    double t = (this->now() - start_time_).seconds() / std::max(time_scale_, 1e-3);
    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "world";
    for (auto & o : obstacles_) {
      auto [dx, dy] = interp(o.keys, t);
      double x = o.base_x + dx;
      double y = o.base_y + dy;
      set_pose("moving_obs_" + std::to_string(o.idx), x, y);

      geometry_msgs::msg::Pose p;
      p.position.x = x;
      p.position.y = y;
      p.position.z = z_height_;
      p.orientation.w = 1.0;
      msg.poses.push_back(p);
    }
    if (!msg.poses.empty()) {
      obstacle_pose_pub_->publish(msg);
    }
  }

  void set_pose(const std::string & name, double x, double y)
  {
    ignition::msgs::Pose req;
    req.set_name(name);
    req.mutable_position()->set_x(x);
    req.mutable_position()->set_y(y);
    req.mutable_position()->set_z(z_height_);
    req.mutable_orientation()->set_x(0.0);
    req.mutable_orientation()->set_y(0.0);
    req.mutable_orientation()->set_z(0.0);
    req.mutable_orientation()->set_w(1.0);

    ignition::msgs::Boolean res;
    bool result = false;
    // Short timeout: at 20 Hz tick rate, occasional drops are fine.
    ign_node_.Request("/world/" + world_name_ + "/set_pose", req, 40, res, result);
  }

  std::string world_name_;
  std::string motion_mode_;
  double      update_rate_;
  double      z_height_;
  double      time_scale_;
  std::vector<ObstacleSpec>   obstacles_;
  rclcpp::Time                start_time_;
  rclcpp::TimerBase::SharedPtr timer_;
  ignition::transport::Node   ign_node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obstacle_pose_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicObstacleNode>());
  rclcpp::shutdown();
  return 0;
}
