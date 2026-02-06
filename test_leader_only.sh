#!/bin/bash

source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

echo "============================================"
echo "Testing Leader Only (SO100)"
echo "============================================"
echo ""

# Check Gazebo
if ! ros2 topic list | grep -q "/so100/joint_states"; then
    echo "ERROR: Gazebo not running!"
    exit 1
fi

echo "✓ Gazebo running"
echo ""

# Check controllers
SO100_ACTIVE=$(ros2 control list_controllers -c /so100/controller_manager 2>/dev/null | grep "active" | wc -l)

if [ "$SO100_ACTIVE" -lt 2 ]; then
    echo "ERROR: SO100 controllers not active!"
    echo "Please run setup script first"
    exit 1
fi

echo "✓ SO100 controllers active"
echo ""

# Run leader only
echo "Starting Leader (SO100)..."
echo "Watch the LEFT arm move to [-0.4, 0.6]"
echo "Press Ctrl+C to stop"
echo ""

ros2 run so100_ioc_pipeline leader_lqr --ros-args \
  -p q1:=10.0 -p q2:=10.0 -p q3:=1.0 -p q4:=1.0 \
  -p q_des_1:=-0.4 -p q_des_2:=0.6

