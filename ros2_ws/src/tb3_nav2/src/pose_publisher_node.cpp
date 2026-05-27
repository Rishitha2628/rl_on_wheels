// pose_publisher_node — broadcasts the map→odom TF so Nav2 sees the robot
// at its true world position.
//
// In a real robot we'd use AMCL to estimate this transform from lidar +
// the static map. In simulation we have a perfect ground-truth pose from
// Ignition's scene broadcaster, so we compute the correction directly:
//
//   T_map_odom = T_map_robot * inv(T_odom_robot)
//
// The DiffDrive plugin already publishes odom→base_footprint TF. With our
// map→odom on top of that, Nav2 gets a correct map→base_footprint chain.

#include <chrono>
#include <cmath>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose_v.pb.h>

using namespace std::chrono_literals;

class PosePublisherNode : public rclcpp::Node {
public:
  PosePublisherNode() : Node("pose_publisher_node") {
    world_name_       = declare_parameter("world_name",       std::string("empty"));
    robot_model_name_ = declare_parameter("robot_model_name", std::string("waffle_pi"));
    publish_rate_hz_  = declare_parameter("publish_rate_hz",  20.0);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { on_odom(msg); });

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

    std::string pose_topic = "/world/" + world_name_ + "/dynamic_pose/info";
    bool ok = ign_node_.Subscribe(pose_topic, &PosePublisherNode::on_ign_pose, this);
    RCLCPP_INFO(get_logger(),
                "PosePublisherNode: world=%s robot=%s ign_pose_topic=%s subscribed=%d",
                world_name_.c_str(), robot_model_name_.c_str(),
                pose_topic.c_str(), ok);

    auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / std::max(1.0, publish_rate_hz_)));
    timer_ = create_wall_timer(period, [this]() { publish_tf(); });
  }

private:
  void on_ign_pose(const ignition::msgs::Pose_V & msg) {
    for (int i = 0; i < msg.pose_size(); ++i) {
      const auto & p = msg.pose(i);
      if (p.name() == robot_model_name_) {
        std::lock_guard<std::mutex> lk(mtx_);
        true_x_   = static_cast<float>(p.position().x());
        true_y_   = static_cast<float>(p.position().y());
        true_yaw_ = quat_yaw(p.orientation().z(), p.orientation().w());
        have_true_ = true;
        return;
      }
    }
  }

  void on_odom(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    odom_x_   = static_cast<float>(msg->pose.pose.position.x);
    odom_y_   = static_cast<float>(msg->pose.pose.position.y);
    odom_yaw_ = quat_yaw(msg->pose.pose.orientation.z,
                         msg->pose.pose.orientation.w);
    have_odom_ = true;
  }

  static float quat_yaw(double qz, double qw) {
    return std::atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz);
  }

  void publish_tf() {
    float tx, ty, tyaw, ox, oy, oyaw;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!have_true_ || !have_odom_) return;
      tx = true_x_; ty = true_y_; tyaw = true_yaw_;
      ox = odom_x_; oy = odom_y_; oyaw = odom_yaw_;
    }

    // T_map_odom = T_map_robot * inv(T_odom_robot).
    // For a 2D rigid transform with translation (x,y) and rotation θ,
    // inverse is:  yaw' = -θ,  trans' = -R(-θ) · (x,y)
    float yaw_diff = tyaw - oyaw;
    float c = std::cos(yaw_diff);
    float s = std::sin(yaw_diff);
    float dx = tx - (c * ox - s * oy);
    float dy = ty - (s * ox + c * oy);

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = "map";
    t.child_frame_id  = "odom";
    t.transform.translation.x = dx;
    t.transform.translation.y = dy;
    t.transform.translation.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw_diff);
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(t);
  }

  std::string world_name_, robot_model_name_;
  double publish_rate_hz_;

  std::mutex mtx_;
  float true_x_ = 0.0f, true_y_ = 0.0f, true_yaw_ = 0.0f;
  float odom_x_ = 0.0f, odom_y_ = 0.0f, odom_yaw_ = 0.0f;
  bool have_true_ = false, have_odom_ = false;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  ignition::transport::Node ign_node_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PosePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
