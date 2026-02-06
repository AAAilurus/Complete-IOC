#!/bin/bash

source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

echo "============================================"
echo "Testing Follower Only (SO101)"
echo "============================================"
echo ""

# Check Gazebo
if ! ros2 topic list | grep -q "/so101/joint_states"; then
    echo "ERROR: Gazebo not running or SO101 not spawned!"
    exit 1
fi

echo "✓ Gazebo running"
echo ""

# Check controllers
SO101_ACTIVE=$(ros2 control list_controllers -c /so101/controller_manager 2>/dev/null | grep "active" | wc -l)

if [ "$SO101_ACTIVE" -lt 2 ]; then
    echo "ERROR: SO101 controllers not active!"
    exit 1
fi

echo "✓ SO101 controllers active"
echo ""

# Check learned Q
if [ ! -f "/root/so100_ws/ioc_results/learned_Q.npz" ]; then
    echo "ERROR: Learned Q not found!"
    exit 1
fi

echo "✓ Learned Q found"
echo ""

# Run follower with VERY conservative control
echo "Starting Follower (SO101) with learned Q..."
echo "Control scale: 0.01 (very conservative)"
echo "Watch the RIGHT arm move to [-0.4, 0.6]"
echo "Press Ctrl+C to stop"
echo ""

ros2 run so100_ioc_pipeline follower_ioc_fixed --ros-args \
  -p q_file:=/root/so100_ws/ioc_results/learned_Q.npz \
  -p control_scale:=0.01 \
  -p log_every:=50 \
  -p log_file:=/root/so100_ws/ioc_results/follower_data.csv \
  -p q_des_1:=-0.4 -p q_des_2:=0.6

