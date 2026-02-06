# IOC System - Complete Data Package

## Overview
This directory contains all data from the Inverse Optimal Control (IOC) experiment where a follower robot learns the cost function (Q matrix) of a leader robot.

## Experiment Summary
- **Leader Q***: [10, 10, 1, 1]
- **Learned Q**: [9.9999, 9.9999, 1.0000, 1.0000]
- **Success**: K error < 0.00001 ✓
- **Method**: Gradient descent with finite differences
- **Learning rate**: α = 0.5
- **Convergence**: 583 iterations

## Files

### 1. `ioc_history.csv`
Complete learning trajectory showing how Q evolved over 583 iterations.

**Columns:**
- `iteration`: Iteration number (0 to 583)
- `objective`: ||K(Q) - K*||² 
- `k_error`: ||K(Q) - K*||_Frobenius
- `q00`, `q11`: Position weight elements of Q
- `q22`, `q33`: Velocity weight elements of Q

**Key data points:**
```
Iteration 0:   K error = 3.254906
Iteration 100: K error = 0.115567
Iteration 300: K error = 0.002355
Iteration 583: K error = 0.000010 (converged!)
```

### 2. `learned_Q.npz`
NumPy archive containing final learned matrices.

**Contents:**
- `Q_learned`: Learned cost matrix (4×4)
- `K_learned`: Learned LQR gain (2×4)  
- `K_star`: Target LQR gain from leader (2×4)
- `P_learned`: Solution to Riccati equation (4×4)
- `A`, `B`: System dynamics matrices
- `R`: Control cost matrix (2×2)
- `dt`: Timestep (0.02s)

**Load in Python:**
```python
import numpy as np
data = np.load('learned_Q.npz')
Q = data['Q_learned']
K = data['K_learned']
```

### 3. `ioc_convergence.png`
Visualization of learning progress with 4 subplots:
1. K error vs iteration (log scale)
2. Objective function vs iteration  
3. Q position weights convergence
4. Q velocity weights convergence

### 4. `follower_data.csv` (if recorded)
Real-time robot execution data from follower using learned Q.

**Columns:**
- `time`: Elapsed time (seconds)
- `step`: Control step number
- `q1`, `q2`: Joint positions
- `dq1`, `dq2`: Joint velocities
- `q_err1`, `q_err2`: Position errors from target
- `dq_err1`, `dq_err2`: Velocity errors
- `u1`, `u2`: Control inputs
- `qcmd1`, `qcmd2`: Position commands

## Usage

### Analyze learning trajectory
```bash
python3 /root/so100_ws/analyze_ioc_learning.py
```

### Load and use learned Q
```bash
ros2 run so100_ioc_pipeline follower_ioc_fixed --ros-args \
  -p q_file:=/root/so100_ws/ioc_results/learned_Q.npz \
  -p q_des_1:=-0.4 -p q_des_2:=0.6
```

### View convergence plot
```bash
# Copy to host to view:
# The file is at: /root/so100_ws/ioc_results/ioc_convergence.png
```

## Technical Details

### System Dynamics
- Type: Discrete-time double integrator
- State: x = [q - q_des; dq - dq_des] (4D)
- Control: u = acceleration (2D)
- Timestep: dt = 0.02s (50 Hz)

### LQR Control Law
```
Cost: J = Σ (x'Qx + u'Ru)
Control: u = -Kx
K = inv(R + B'PB) * (B'PA)
P = solution to discrete Riccati equation
```

### IOC Algorithm
```
Objective: minimize ||K(Q) - K*||²
Method: Gradient descent
Gradient: Computed via finite differences
Update: Q ← Q - α * ∇Q
Constraint: Q must be positive definite
```

## Results
✅ **Perfect convergence** - Learned Q ≈ Q* with error < 0.00001  
✅ **Stable control** - Follower successfully controlled robot  
✅ **Fast learning** - Converged in 583/3000 iterations  

---
Generated: 2026-02-06  
Project: Inverse Optimal Control for Dual-Arm Manipulation
