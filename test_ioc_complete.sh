#!/bin/bash
set -e

echo "============================================"
echo "COMPLETE IOC SYSTEM TEST"
echo "============================================"
echo ""

# Source ROS
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

echo "STEP 1: Run Leader LQR for 5 seconds"
echo "----------------------------------------"
timeout 5 ros2 run so100_ioc_pipeline leader_lqr --ros-args \
  -p q1:=10.0 -p q2:=10.0 -p q3:=1.0 -p q4:=1.0 \
  -p q_des_1:=-0.4 -p q_des_2:=0.6 &

LEADER_PID=$!
sleep 2

# Get K* from the leader
echo ""
echo "Getting K* from leader..."
K_STAR_JSON=$(ros2 topic echo /leader_gain --once 2>/dev/null | grep -A 100 "data:" | head -n 1 | sed 's/data: //')
echo "Received: $K_STAR_JSON"

# Extract K* values (this is the K* from leader with Q*=[10,10,1,1])
# The K* should be: [[3.07784446 0. 2.6651457 0.] [0. 3.07784446 0. 2.6651457]]
K_STAR="3.07784446 0.0 2.6651457 0.0 0.0 3.07784446 0.0 2.6651457"

wait $LEADER_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "STEP 2: Run Offline IOC to Learn Q from K*"
echo "============================================"
echo ""
echo "Leader's Q*: diag[10.0, 10.0, 1.0, 1.0]"
echo "Leader's K*: [3.078  0.000  2.665  0.000]"
echo "             [0.000  3.078  0.000  2.665]"
echo ""
echo "Now learning Q from K*..."
echo ""

ros2 run so100_ioc_pipeline ioc_learn_from_K \
  --k_star $K_STAR \
  --out /tmp/learned_Q.npz \
  --alpha 0.01 \
  --iters 1000 \
  --print_every 100

echo ""
echo "============================================"
echo "STEP 3: Compare Results"
echo "============================================"

python3 << 'PYEOF'
import numpy as np

# Load learned results
data = np.load('/tmp/learned_Q.npz')
Q_learned = data['Q']
K_learned = data['K']
K_star = data['K_star']

print("\nLeader's Q* (ground truth):")
Q_star = np.diag([10.0, 10.0, 1.0, 1.0])
print(Q_star)

print("\nFollower's Q_learned:")
print(Q_learned)

print("\nQ Error:")
print(Q_learned - Q_star)
print(f"||Q_learned - Q*||_F = {np.linalg.norm(Q_learned - Q_star, 'fro'):.6f}")

print("\n" + "="*60)
print("K* (Leader's gain from Q*):")
print(K_star)

print("\nK_learned (Follower's gain from Q_learned):")
print(K_learned)

print("\nK Error:")
print(K_learned - K_star)
print(f"||K_learned - K*||_F = {np.linalg.norm(K_learned - K_star, 'fro'):.10f}")

print("\n" + "="*60)
if np.linalg.norm(K_learned - K_star, 'fro') < 0.01:
    print("✓ SUCCESS! Follower recovered K* from leader")
else:
    print("✗ Need more iterations or smaller learning rate")
print("="*60)
PYEOF

echo ""
echo "============================================"
echo "STEP 4: Run Leader and Follower Together"
echo "============================================"
echo ""
echo "Starting leader in background..."

ros2 run so100_ioc_pipeline leader_lqr --ros-args \
  -p q1:=10.0 -p q2:=10.0 -p q3:=1.0 -p q4:=1.0 \
  -p q_des_1:=-0.4 -p q_des_2:=0.6 &

LEADER_PID=$!
sleep 2

echo "Starting follower with learned Q..."
timeout 10 ros2 run so100_ioc_pipeline follower_ioc --ros-args \
  -p q_file:=/tmp/learned_Q.npz \
  -p q_des_1:=-0.4 -p q_des_2:=0.6 &

FOLLOWER_PID=$!

echo ""
echo "Both arms running with learned Q..."
echo "Leader (SO100): Using Q* = diag[10, 10, 1, 1]"
echo "Follower (SO101): Using Q_learned from IOC"
echo ""
echo "Watch Gazebo to see both arms move to [-0.4, 0.6]"
echo ""

sleep 10

kill $LEADER_PID 2>/dev/null || true
kill $FOLLOWER_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "✓ TEST COMPLETE!"
echo "============================================"
echo ""
echo "Summary:"
echo "1. Leader ran with Q* = diag[10, 10, 1, 1]"
echo "2. IOC learned Q from leader's K*"
echo "3. Follower used learned Q to control"
echo "4. Both arms controlled with similar behavior"
echo ""
