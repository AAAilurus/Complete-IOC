#!/usr/bin/env python3
"""
Discrete-Time IOC with Exact Gradient (Python version of MATLAB code)
Learns Q from K* using discrete Lyapunov equations
"""
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
import matplotlib.pyplot as plt

def dlqr(A, B, Q, R):
    """Discrete-time LQR (like MATLAB's dlqr)"""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def exact_grad_diagQ_dt(Ad, Bd, R, K, P, K_star):
    """
    Exact gradient using discrete Lyapunov equation
    
    For each diagonal element Q[i,i]:
        Acl' dP Acl - dP + E_i = 0  (discrete Lyapunov)
        dK = (R + Bd'P Bd)^(-1) Bd' dP Acl
        grad[i] = 2 * <K-K*, dK>
    """
    n = Ad.shape[0]
    
    E = K - K_star  # Error matrix (m x n)
    Acl = Ad - Bd @ K  # Closed-loop A
    S = R + Bd.T @ P @ Bd  # m x m
    
    g = np.zeros(n)
    
    for i in range(n):
        # Create E_i (only diagonal element i)
        Ei = np.zeros((n, n))
        Ei[i, i] = 1.0
        
        # Solve discrete Lyapunov: Acl' dP Acl - dP + E_i = 0
        dP = solve_discrete_lyapunov(Acl.T, Ei)
        dP = 0.5 * (dP + dP.T)
        
        # Compute dK
        dK = np.linalg.inv(S) @ (Bd.T @ dP @ Acl)
        
        # Gradient: 2 * Frobenius inner product
        g[i] = 2 * np.sum(E * dK)
    
    return g

def proj_psd_floor(Q, floor_val):
    """Project Q to be positive definite with minimum eigenvalue floor_val"""
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.maximum(eigvals, floor_val)
    Q_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (Q_proj + Q_proj.T)

# ============================================
# MAIN IOC LEARNING
# ============================================

print("="*70)
print("DISCRETE-TIME IOC LEARNING (Python)")
print("="*70)

# 1. System parameters
Ts = 0.02  # Sampling time (20ms = 50Hz)
print(f"\nSampling time Ts = {Ts:.4f} s ({1/Ts:.1f} Hz)")

# 2. Discrete-time double integrator (ZOH discretization)
Ad = np.array([[1, 0, Ts, 0],
               [0, 1, 0, Ts],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

Bd = np.array([[0.5*Ts**2, 0],
               [0, 0.5*Ts**2],
               [Ts, 0],
               [0, Ts]])

n = Ad.shape[0]
m = Bd.shape[1]

print(f"System: {n} states, {m} controls")
print(f"Controllability rank: {np.linalg.matrix_rank(np.hstack([Bd, Ad@Bd, Ad@Ad@Bd, Ad@Ad@Ad@Bd]))}/{n}")

# 3. Leader's parameters
Q_star = np.diag([100.0, 100.0, 10.0, 10.0])  # True Q
R_star = np.diag([0.5, 0.5])  # Control cost

# Compute leader's K*
K_star, P_star = dlqr(Ad, Bd, Q_star, R_star)

print("\n" + "="*70)
print("LEADER (Expert)")
print("="*70)
print("Q* (true):")
print(Q_star)
print("\nK* (from dlqr):")
print(K_star)

# 4. IOC configuration
alphaQ = 0.5  # Learning rate
max_iters = 3000
tol_K = 1e-4
pd_floor = 1e-6

# Initial Q (start from different value!)
Q0 = np.diag([10.0, 10.0, 1.0, 1.0])  # NOT same as Q*

print("\n" + "="*70)
print("IOC CONFIGURATION")
print("="*70)
print(f"Initial Q: {np.diag(Q0)}")
print(f"Learning rate: {alphaQ}")
print(f"Max iterations: {max_iters}")
print(f"Convergence tolerance: {tol_K}")

# 5. Run IOC learning
print("\n" + "="*70)
print("LEARNING...")
print("="*70)

Q = Q0.copy()
hist_K_err = []
hist_Q_err = []
hist_Q_diag = []
hist_grad_norm = []

for iteration in range(max_iters):
    # Compute K from current Q
    K, P = dlqr(Ad, Bd, Q, R_star)
    
    # Compute errors
    K_err = np.linalg.norm(K - K_star, 'fro')
    Q_err = np.linalg.norm(Q - Q_star, 'fro')
    
    hist_K_err.append(K_err)
    hist_Q_err.append(Q_err)
    hist_Q_diag.append(np.diag(Q).copy())
    
    # Print progress
    if iteration % 100 == 0 or iteration < 10:
        print(f"Iter {iteration:4d}: ||K-K*||={K_err:.6f}, Q={np.diag(Q)}")
    
    # Check convergence
    if K_err <= tol_K:
        print(f"\n✓ CONVERGED at iteration {iteration}!")
        break
    
    # Compute exact gradient using discrete Lyapunov
    grad = exact_grad_diagQ_dt(Ad, Bd, R_star, K, P, K_star)
    hist_grad_norm.append(np.linalg.norm(grad))
    
    # Update Q (only diagonal)
    Q_new = Q - alphaQ * np.diag(grad)
    Q_new = proj_psd_floor(Q_new, pd_floor)
    
    Q = Q_new

# Final K with learned Q
K_learned, P_learned = dlqr(Ad, Bd, Q, R_star)

# 6. Results
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print("Target Q*:")
print(Q_star)
print("\nLearned Q:")
print(Q)
print("\nTarget K*:")
print(K_star)
print("\nLearned K:")
print(K_learned)

K_err_final = np.linalg.norm(K_learned - K_star, 'fro')
Q_err_final = np.linalg.norm(Q - Q_star, 'fro')

print(f"\nFinal ||K-K*|| = {K_err_final:.10f}")
print(f"Final ||Q-Q*|| = {Q_err_final:.10f}")
print(f"K error: {100*K_err_final/np.linalg.norm(K_star, 'fro'):.4f}%")
print(f"Q error: {100*Q_err_final/np.linalg.norm(Q_star, 'fro'):.4f}%")

# 7. Save results
np.savez('ioc_results/learned_Q_discrete.npz',
         Q=Q, K=K_learned, K_star=K_star, Q_star=Q_star,
         Ad=Ad, Bd=Bd, R=R_star, Ts=Ts,
         hist_K_err=hist_K_err,
         hist_Q_err=hist_Q_err,
         hist_Q_diag=hist_Q_diag)

print("\n✓ Saved to: ioc_results/learned_Q_discrete.npz")

# 8. Plot convergence
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogy(hist_K_err)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||K - K*||_F')
plt.title('K Error Convergence')

plt.subplot(2, 2, 2)
plt.semilogy(hist_Q_err)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||Q - Q*||_F')
plt.title('Q Error Convergence')

plt.subplot(2, 2, 3)
hist_Q_diag = np.array(hist_Q_diag)
for i in range(4):
    plt.plot(hist_Q_diag[:, i], label=f'Q[{i},{i}]')
    plt.axhline(Q_star[i, i], linestyle='--', alpha=0.5)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Q diagonal values')
plt.title('Q Convergence')
plt.legend()

plt.subplot(2, 2, 4)
plt.semilogy(hist_grad_norm)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('||gradient||')
plt.title('Gradient Norm')

plt.tight_layout()
plt.savefig('ioc_results/discrete_ioc_convergence.png', dpi=150)
print("✓ Saved plot to: ioc_results/discrete_ioc_convergence.png")

print("\n" + "="*70)
print("DONE! Use learned Q for follower robot.")
print("="*70)
