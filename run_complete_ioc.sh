#!/bin/bash
# ==============================================================================
# COMPLETE AUTOMATED IOC SYSTEM
# Run leader → Learn Q → Run follower → Analyze
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters (can be overridden by command line arguments)
Q1=${1:-100.0}
Q2=${2:-100.0}
Q3=${3:-10.0}
Q4=${4:-10.0}
R1=${5:-0.5}
R2=${6:-0.5}
Q_DES_1=${7:-0.5}
Q_DES_2=${8:--0.6}
LEADER_TIME=${9:-30}
FOLLOWER_TIME=${10:-30}

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}                 COMPLETE AUTOMATED IOC SYSTEM${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Parameters:${NC}"
echo "  Q_star = [$Q1, $Q2, $Q3, $Q4]"
echo "  R      = [$R1, $R2]"
echo "  Target = [$Q_DES_1, $Q_DES_2] rad"
echo "  Leader time:   ${LEADER_TIME}s"
echo "  Follower time: ${FOLLOWER_TIME}s"
echo ""

# Setup
cd /root/so100_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
mkdir -p ioc_results

# ==============================================================================
# PHASE 1: START DATA LOGGER
# ==============================================================================
echo -e "${YELLOW}[PHASE 1] Starting data logger...${NC}"
python3 unified_ioc_logger.py > ioc_results/logger.log 2>&1 &
LOGGER_PID=$!
sleep 2
echo -e "${GREEN}✓ Logger started (PID: $LOGGER_PID)${NC}"

# ==============================================================================
# PHASE 2: RUN LEADER
# ==============================================================================
echo ""
echo -e "${YELLOW}[PHASE 2] Running LEADER robot with Q* = [$Q1, $Q2, $Q3, $Q4]...${NC}"
echo "  This will take ${LEADER_TIME} seconds..."

timeout ${LEADER_TIME}s ros2 run so100_lqr lqr_with_Q --ros-args \
  -p state_topic:=/so100/joint_states \
  -p cmd_topic:=/so100/arm_position_controller/commands \
  -p rate_hz:=50.0 \
  -p q_des:="[$Q_DES_1, $Q_DES_2]" \
  -p Q_diag:="[$Q1, $Q2, $Q3, $Q4]" \
  -p R_diag:="[$R1, $R2]" \
  -p control_scale:=0.01 \
  > ioc_results/leader.log 2>&1 || true

echo -e "${GREEN}✓ Leader finished${NC}"
sleep 2

# ==============================================================================
# PHASE 3: IOC LEARNING
# ==============================================================================
echo ""
echo -e "${YELLOW}[PHASE 3] Learning Q from leader's K*...${NC}"

# Create IOC learning script with current parameters
cat > /tmp/ioc_learn_now.py << IOCEOF
#!/usr/bin/env python3
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def exact_grad_diagQ_dt(Ad, Bd, R, K, P, K_star):
    n = Ad.shape[0]
    E = K - K_star
    Acl = Ad - Bd @ K
    S = R + Bd.T @ P @ Bd
    g = np.zeros(n)
    for i in range(n):
        Ei = np.zeros((n, n))
        Ei[i, i] = 1.0
        dP = solve_discrete_lyapunov(Acl.T, Ei)
        dP = 0.5 * (dP + dP.T)
        dK = np.linalg.inv(S) @ (Bd.T @ dP @ Acl)
        g[i] = 2 * np.sum(E * dK)
    return g

def proj_psd_floor(Q, floor_val):
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.maximum(eigvals, floor_val)
    Q_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (Q_proj + Q_proj.T)

# System
Ts = 0.02
Ad = np.array([[1, 0, Ts, 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]])
Bd = np.array([[0.5*Ts**2, 0], [0, 0.5*Ts**2], [Ts, 0], [0, Ts]])

# Leader's parameters
Q_star = np.diag([$Q1, $Q2, $Q3, $Q4])
R_star = np.diag([$R1, $R2])
K_star, P_star = dlqr(Ad, Bd, Q_star, R_star)

print("Leader K*:")
print(K_star)

# IOC Learning
Q = np.diag([10.0, 10.0, 1.0, 1.0])
alphaQ = 0.5
max_iters = 3000
tol_K = 1e-4

print("\nLearning Q...")
for iteration in range(max_iters):
    K, P = dlqr(Ad, Bd, Q, R_star)
    K_err = np.linalg.norm(K - K_star, 'fro')
    
    if iteration % 500 == 0:
        print(f"Iter {iteration:4d}: ||K-K*||={K_err:.6f}")
    
    if K_err <= tol_K:
        print(f"\n✓ CONVERGED at iteration {iteration}!")
        break
    
    grad = exact_grad_diagQ_dt(Ad, Bd, R_star, K, P, K_star)
    Q = Q - alphaQ * np.diag(grad)
    Q = proj_psd_floor(Q, 1e-6)

K_learned, P_learned = dlqr(Ad, Bd, Q, R_star)

print("\nLearned Q:")
print(np.diag(Q))
print("\nLearned K:")
print(K_learned)
print(f"\n||K-K*|| = {np.linalg.norm(K_learned - K_star, 'fro'):.10f}")

# Save
np.savez('/root/so100_ws/ioc_results/learned_Q_discrete.npz',
         Q=Q, K=K_learned, K_star=K_star, Q_star=Q_star,
         Ad=Ad, Bd=Bd, R=R_star, Ts=Ts)

print("\n✓ Saved learned Q")
IOCEOF

python3 /tmp/ioc_learn_now.py > ioc_results/ioc_learning.log 2>&1
echo -e "${GREEN}✓ IOC learning complete${NC}"

# Extract learned Q
LEARNED_Q=$(python3 << PYEOF
import numpy as np
data = np.load('/root/so100_ws/ioc_results/learned_Q_discrete.npz')
Q = data['Q']
print(','.join([str(x) for x in np.diag(Q)]))
PYEOF
)

echo "  Learned Q: [$LEARNED_Q]"
sleep 2

# ==============================================================================
# PHASE 4: RUN FOLLOWER
# ==============================================================================
echo ""
echo -e "${YELLOW}[PHASE 4] Running FOLLOWER robot with learned Q...${NC}"
echo "  This will take ${FOLLOWER_TIME} seconds..."

timeout ${FOLLOWER_TIME}s ros2 run so100_lqr lqr_with_Q --ros-args \
  -p state_topic:=/so101/joint_states \
  -p cmd_topic:=/so101/arm_position_controller/commands \
  -p rate_hz:=50.0 \
  -p q_des:="[$Q_DES_1, $Q_DES_2]" \
  -p Q_diag:="[$LEARNED_Q]" \
  -p R_diag:="[$R1, $R2]" \
  -p control_scale:=0.01 \
  > ioc_results/follower.log 2>&1 || true

echo -e "${GREEN}✓ Follower finished${NC}"
sleep 2

# ==============================================================================
# PHASE 5: STOP LOGGER AND SAVE
# ==============================================================================
echo ""
echo -e "${YELLOW}[PHASE 5] Stopping logger and saving data...${NC}"
kill -INT $LOGGER_PID 2>/dev/null || true
sleep 3
echo -e "${GREEN}✓ Data saved${NC}"

# ==============================================================================
# PHASE 6: RESULTS
# ==============================================================================
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}                           RESULTS${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Show CSV header
echo -e "${GREEN}Data file header:${NC}"
head -25 ioc_results/unified_data.csv

echo ""
echo -e "${GREEN}Final positions:${NC}"
tail -5 ioc_results/unified_data.csv

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}✓ COMPLETE! All data saved to: ioc_results/unified_data.csv${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Files generated:"
echo "  - ioc_results/unified_data.csv         (trajectory data)"
echo "  - ioc_results/learned_Q_discrete.npz   (learned matrices)"
echo "  - ioc_results/leader.log               (leader output)"
echo "  - ioc_results/follower.log             (follower output)"
echo "  - ioc_results/ioc_learning.log         (learning output)"
echo ""
