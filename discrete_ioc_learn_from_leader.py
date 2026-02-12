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

print("="*70)
print("DISCRETE-TIME IOC LEARNING")
print("="*70)

# System (discrete-time, 50Hz)
Ts = 0.02
Ad = np.array([[1, 0, Ts, 0],
               [0, 1, 0, Ts],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

Bd = np.array([[0.5*Ts**2, 0],
               [0, 0.5*Ts**2],
               [Ts, 0],
               [0, Ts]])

# Leader's parameters (PASTE YOUR OBSERVED K* HERE!)
K_star = np.array([[13.19308667, 0.0, 6.6175469, 0.0],
                   [0.0, 13.19308667, 0.0, 6.6175469]])

R_star = np.diag([0.5, 0.5])
Q_star = np.diag([100.0, 100.0, 10.0, 10.0])  # True Q (for reference)

print("\nObserved K* from leader:")
print(K_star)
print("\nTarget Q* (reference):")
print(Q_star)

# IOC Learning
Q = np.diag([10.0, 10.0, 1.0, 1.0])  # Start different!
alphaQ = 0.5
max_iters = 3000
tol_K = 1e-4

print(f"\nInitial Q: {np.diag(Q)}")
print(f"\nLearning...")

for iteration in range(max_iters):
    K, P = dlqr(Ad, Bd, Q, R_star)
    K_err = np.linalg.norm(K - K_star, 'fro')
    
    if iteration % 100 == 0 or iteration < 10:
        print(f"Iter {iteration:4d}: ||K-K*||={K_err:.6f}, Q={np.diag(Q)}")
    
    if K_err <= tol_K:
        print(f"\n✓ CONVERGED at iteration {iteration}!")
        break
    
    grad = exact_grad_diagQ_dt(Ad, Bd, R_star, K, P, K_star)
    Q = Q - alphaQ * np.diag(grad)
    Q = proj_psd_floor(Q, 1e-6)

K_learned, P_learned = dlqr(Ad, Bd, Q, R_star)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print("Learned Q:")
print(Q)
print("\nLearned K:")
print(K_learned)
print(f"\n||K-K*|| = {np.linalg.norm(K_learned - K_star, 'fro'):.10f}")
print(f"||Q-Q*|| = {np.linalg.norm(Q - Q_star, 'fro'):.10f}")

# Save
np.savez('ioc_results/learned_Q_discrete.npz',
         Q=Q, K=K_learned, K_star=K_star, Q_star=Q_star,
         Ad=Ad, Bd=Bd, R=R_star, Ts=Ts)

print("\n✓ Saved to: ioc_results/learned_Q_discrete.npz")
print("\nLearned Q diagonal for follower:")
print(list(np.diag(Q)))
