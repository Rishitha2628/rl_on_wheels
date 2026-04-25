FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV TURTLEBOT3_MODEL=waffle_pi

# Ignition Fortress + TurtleBot3 ROS2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-interfaces \
    libignition-gazebo6-dev \
    libignition-transport11-dev \
    libignition-msgs8-dev \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-description \
    ros-humble-turtlebot3-bringup \
    ros-humble-visualization-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-geometry-msgs \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Python RL dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Build ROS2 workspace
WORKDIR /ros2_ws
COPY ros2_ws/src /ros2_ws/src
RUN /bin/bash -c "\
    source /opt/ros/humble/setup.bash && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
    "

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
