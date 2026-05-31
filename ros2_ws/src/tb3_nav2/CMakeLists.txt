cmake_minimum_required(VERSION 3.8)
project(tb3_nav2)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ignition-transport11 REQUIRED)
find_package(ignition-msgs8 REQUIRED)

# ── pose_publisher_node ───────────────────────────────────────────────────────
add_executable(pose_publisher_node src/pose_publisher_node.cpp)
ament_target_dependencies(pose_publisher_node
  rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(pose_publisher_node
  ignition-transport11::core
  ignition-msgs8::core)

# ── goal_forwarder_node ───────────────────────────────────────────────────────
add_executable(goal_forwarder_node src/goal_forwarder_node.cpp)
ament_target_dependencies(goal_forwarder_node
  rclcpp rclcpp_action geometry_msgs nav2_msgs std_msgs)

# ── install ───────────────────────────────────────────────────────────────────
install(TARGETS
  pose_publisher_node
  goal_forwarder_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
