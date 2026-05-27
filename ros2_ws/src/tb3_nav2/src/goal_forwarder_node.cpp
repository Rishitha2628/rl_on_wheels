// goal_forwarder_node — bridges /goal_pose topic to Nav2's /navigate_to_pose
// action. Each new PoseStamped on /goal_pose cancels any in-flight nav goal
// and submits the new one to Nav2.

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

using namespace std::chrono_literals;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandle     = rclcpp_action::ClientGoalHandle<NavigateToPose>;


class GoalForwarderNode : public rclcpp::Node {
public:
  GoalForwarderNode() : Node("goal_forwarder_node") {
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose");

    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
        on_goal(msg);
      });

    RCLCPP_INFO(get_logger(),
                "GoalForwarderNode ready — waiting for /goal_pose messages");
  }

private:
  void on_goal(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
    if (!nav_client_->wait_for_action_server(1s)) {
      RCLCPP_WARN(get_logger(),
                  "/navigate_to_pose action server not available yet — "
                  "dropping this goal");
      return;
    }

    // Cancel any in-flight goal so the new one takes over cleanly.
    if (current_handle_) {
      nav_client_->async_cancel_goal(current_handle_);
    }

    NavigateToPose::Goal goal_msg;
    goal_msg.pose = *msg;
    // reset_node publishes in world frame which we expose to Nav2 as "map".
    goal_msg.pose.header.frame_id = "map";
    goal_msg.pose.header.stamp    = now();

    auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    opts.goal_response_callback =
      [this](std::shared_ptr<GoalHandle> h) {
        if (!h) {
          RCLCPP_WARN(get_logger(), "Nav2 rejected the goal");
        } else {
          current_handle_ = h;
        }
      };
    opts.result_callback = [this](const GoalHandle::WrappedResult & r) {
      switch (r.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
          RCLCPP_INFO(get_logger(), "[nav2] goal SUCCEEDED");
          break;
        case rclcpp_action::ResultCode::ABORTED:
          RCLCPP_WARN(get_logger(), "[nav2] goal ABORTED");
          break;
        case rclcpp_action::ResultCode::CANCELED:
          RCLCPP_INFO(get_logger(), "[nav2] goal CANCELED");
          break;
        default:
          RCLCPP_WARN(get_logger(), "[nav2] goal unknown result");
      }
      current_handle_.reset();
    };

    nav_client_->async_send_goal(goal_msg, opts);
    RCLCPP_INFO(get_logger(),
                "Forwarded goal (%.2f, %.2f) to Nav2",
                msg->pose.position.x, msg->pose.position.y);
  }

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  std::shared_ptr<GoalHandle> current_handle_;
};


int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GoalForwarderNode>());
  rclcpp::shutdown();
  return 0;
}
