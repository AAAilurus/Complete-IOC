#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

echo "============================================"
echo "IOC Demo in Gazebo"
echo "============================================"
echo ""

# Check if Gazebo is running
if ! ros2 topic list | grep -q "/so100/joint_states"; then
    echo "ERROR: Gazebo not running or SO100 not spawned!"
    echo "Please start Gazebo first:"
    echo "  ros2 launch so100_2dof_bringup gz_dual_2dof.launch.py"
    exit 1
fi

if ! ros2 topic list | grep -q "/so101/joint_states"; then
    echo "ERROR: SO101 not spawned!"
    echo "Please start Gazebo with both robots"
    exit 1
fi

echo "✓ Gazebo running with both robots"
echo ""

# Check controllers
echo "Checking controllers..."
SO100_ACTIVE=$(ros2 control list_controllers -c /so100/controller_manager 2>/dev/null | grep "active" | wc -l)
SO101_ACTIVE=$(ros2 control list_controllers -c /so101/controller_manager 2>/dev/null | grep "active" | wc -l)

if [ "$SO100_ACTIVE" -lt 2 ] || [ "$SO101_ACTIVE" -lt 2 ]; then
    echo "Setting up controllers..."
    /root/so100_ws/setup_controllers.sh
    sleep 3
fi

echo "✓ Controllers active"
echo ""

# Verify learned Q exists
if [ ! -f "/root/so100_ws/ioc_results/learned_Q.npz" ]; then
    echo "ERROR: Learned Q not found!"
    echo "Please run IOC learning first:"
    echo "  ros2 run so100_ioc_pipeline ioc_learn_from_K ..."
    exit 1
fi

echo "✓ Learned Q found"
echo ""

# Start leader
echo "============================================"
echo "Starting Leader (SO100) - LEFT ARM"
echo "============================================"
echo "  Q* = [10, 10, 1, 1]"
echo ""

ros2 run so100_ioc_pipeline leader_lqr --ros-args \
  -p q1:=10.0 -p q2:=10.0 -p q3:=1.0 -p q4:=1.0 \
  -p q_des_1:=-0.4 -p q_des_2:=0.6 &

LEADER_PID=$!
sleep 3

# Start follower
echo ""
echo "============================================"
echo "Starting Follower (SO101) - RIGHT ARM"
echo "============================================"
echo "  Q_learned ≈ [10, 10, 1, 1]"
echo "  Control scale: 0.02 (very conservative for stability)"
echo ""

ros2 run so100_ioc_pipeline follower_ioc_fixed --ros-args \
  -p q_file:=/root/so100_ws/ioc_results/learned_Q.npz \
  -p control_scale:=0.02 \
  -p log_every:=50 \
  -p log_file:=/root/so100_ws/ioc_results/follower_data.csv \
  -p q_des_1:=-0.4 -p q_des_2:=0.6 &

FOLLOWER_PID=$!

echo ""
echo "============================================"
echo "✓ Both arms running in Gazebo!"
echo "============================================"
echo ""
echo "Leader (SO100):  LEFT arm  - using Q* = [10,10,1,1]"
echo "Follower (SO101): RIGHT arm - using Q_learned ≈ [10,10,1,1]"
echo ""
echo "Both should move to position [-0.4, 0.6]"
echo "Watch Gazebo to see coordinated motion!"
echo ""
echo "Recording follower data every 50 steps..."
echo "Press Ctrl+C to stop"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping..."
    kill $LEADER_PID $FOLLOWER_PID 2>/dev/null
    sleep 1
    
    if [ -f "/root/so100_ws/ioc_results/follower_data.csv" ]; then
        LINES=$(wc -l < /root/so100_ws/ioc_results/follower_data.csv)
        echo "✓ Logged $LINES data points"
        echo "  View: head -n 20 /root/so100_ws/ioc_results/follower_data.csv"
    fi
    
    exit 0
}

trap cleanup INT

wait
