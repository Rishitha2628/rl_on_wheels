#!/usr/bin/env bash
STAGE=${STAGE:-5}
export STAGE
# Render the overhead + isometric views of the stage-4 arena used in the paper.
#
#   docker start sim && scripts/render_arena.sh
#
# Writes paper/figures/arena_render_{top,iso}.png. scripts/make_paper_figs.py
# then overlays the obstacle paths onto the top view to produce arena.pdf.
#
# Ignition Fortress ignores the camera sensor's <save> element, so the frames
# are captured off the published image topic through ros_gz_bridge instead.
set -euo pipefail
cd "$(dirname "$0")/.."
CONTAINER=${CONTAINER:-sim}

python3 - <<'PY'
import os
import os
STAGE = os.environ.get("STAGE", "5")
src = f"ros2_ws/src/tb3_rl_bridge/worlds/tb3_stage{STAGE}.sdf"
s = open(src).read()
def cam(name, pose, fov, w, h):
    return f'''
    <model name="render_{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        <sensor name="{name}" type="camera">
          <topic>{name}</topic>
          <camera>
            <horizontal_fov>{fov}</horizontal_fov>
            <image><width>{w}</width><height>{h}</height></image>
            <clip><near>0.1</near><far>100</far></clip>
          </camera>
          <always_on>1</always_on><update_rate>5</update_rate>
        </sensor>
      </link>
    </model>
'''
# The top camera's pose and FOV are baked into the ground-plane mapping in
# scripts/make_paper_figs.py (RENDER_H_FOV, RENDER_CAM_Z). Change them together.
# Ignition draws its floor grid as a GUI overlay, which a camera sensor does not
# see, so the figure renders without it. These thin flat boxes put a 1 m grid
# into the scene itself, matching how the arena looks in the viewer.
grid = "".join(f"""
    <model name="figure_grid_{ax}_{i}">
      <static>true</static>
      <pose>{0 if ax=='x' else v} {v if ax=='x' else 0} 0.004 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <cast_shadows>false</cast_shadows>
          <geometry><box><size>{'11 0.02 0.001' if ax=='x' else '0.02 11 0.001'}</size></box></geometry>
          <material>
            <ambient>0.62 0.62 0.62 1</ambient>
            <diffuse>0.62 0.62 0.62 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
""" for ax in ("x", "y") for i, v in enumerate(range(-5, 6)))

# Goal placement per stage. Both are chosen so a wall stands between the robot
# spawn (-0.7, 0) and the goal, rather than a clear straight line -- the figure
# should not suggest the task is a straight run.
GOAL_XY = {"5": "1.5 0.55", "6": "2.5 1.8"}.get(STAGE, "1.5 0.55")

goal = """
    <model name="figure_goal_marker">
      <static>true</static>
      <pose>""" + GOAL_XY + """ 0.12 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <geometry><sphere><radius>0.12</radius></sphere></geometry>
          <material>
            <ambient>0.9 0.1 0.1 1</ambient>
            <diffuse>0.9 0.1 0.1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
block = ("\n    <!-- Offscreen cameras and goal marker for the paper figure only. -->\n"
         + grid + goal
         + cam("cam_iso", "7.0 -7.0 6.2 0 0.56 2.3562", 0.95, 1800, 1350)
         + cam("cam_top", "0 0 11.0 0 1.5708 1.5708", 0.78, 1500, 1500)
         + cam("cam_tilt", "0 -5.0 10.5 0 1.126 1.5708", 0.80, 1700, 1400)
         + "\n  </world>")
assert s.count("</world>") == 1
os.makedirs("/tmp/arena_render", exist_ok=True)
open("/tmp/arena_render/world.sdf", "w").write(s.replace("</world>", block))
PY

docker cp /tmp/arena_render/world.sdf "$CONTAINER":/tmp/stage5_render.sdf
docker exec "$CONTAINER" mkdir -p /logs/render
docker exec "$CONTAINER" bash -c '
set +u; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; set -u
export IGN_PARTITION=renderonly ROS_DOMAIN_ID=77
ign gazebo -s -r -v 1 /tmp/stage5_render.sdf > /logs/render/gz.log 2>&1 & GZ=$!
sleep 18
ros2 run ros_gz_bridge parameter_bridge \
  /cam_iso@sensor_msgs/msg/Image[ignition.msgs.Image \
  /cam_top@sensor_msgs/msg/Image[ignition.msgs.Image \
  /cam_tilt@sensor_msgs/msg/Image[ignition.msgs.Image > /logs/render/bridge.log 2>&1 & BR=$!
sleep 10
python3 - <<PY
import time, rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PImage
rclpy.init(); n = Node("grab"); got = {}
def mk(t):
    def cb(m):
        if t in got: return
        a = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
        if m.encoding in ("bgr8", "bgra8"): a = a[:, :, ::-1][:, :, :3]
        PImage.fromarray(a[:, :, :3]).save(f"/logs/render/{t}.png"); got[t] = True
    return cb
for t in ("cam_iso", "cam_top", "cam_tilt"): n.create_subscription(Image, "/" + t, mk(t), 10)
t0 = time.time()
while len(got) < 3 and time.time() - t0 < 60: rclpy.spin_once(n, timeout_sec=0.5)
print("captured:", sorted(got)); rclpy.shutdown()
PY
kill -TERM $BR $GZ 2>/dev/null; sleep 2; kill -9 $BR $GZ 2>/dev/null; true'

docker cp "$CONTAINER":/logs/render/cam_top.png paper/figures/arena_render_top${STAGE}.png
docker cp "$CONTAINER":/logs/render/cam_iso.png paper/figures/arena_render_iso${STAGE}.png
docker cp "$CONTAINER":/logs/render/cam_tilt.png paper/figures/arena_render_tilt${STAGE}.png
chmod 644 paper/figures/arena_render_*.png
docker exec "$CONTAINER" rm -rf /logs/render
echo "wrote paper/figures/arena_render_{top,iso}.png"
