#!/usr/bin/env python3
"""
Discrete-time IOC: Learn Q matrix from expert demonstrations.
Minimizes: ||K_learned - K_expert||_F^2
Uses batch learning with gradient descent via FINITE DIFFERENCES.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import linalg
import control


def parse_args():
    parser = argparse.ArgumentParser(description='Fit Q matrix via discrete-time IOC')
    parser.add_argument('--csv', required=True, help='Clean CSV with expert data')
    parser.add_argument('--out', default='/tmp/ioc_Q.npz', help='Output npz file')
    parser.add_argument('--rate', type=float, default=50.0, help='Control rate (Hz)')
    parser.add_argument('--q_des', type=float, nargs=2, required=True, help='Desired position')
    parser.add_argument('--dq_des', type=float, nargs=2, required=True, help='Desired velocity')
    parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--iters', type=int, default=2000, help='Max iterations')
    parser.add_argument('--pd_floor', type=float, default=1e-6, help='Min eigenvalue for Q')
    parser.add_argument('--ridge', type=float, default=1e-6, help='Ridge regularization')
    parser.add_argument('--print_every', type=int, default=200, help='Print interval')
    parser.add_argument('--grad_eps', type=float, default=1e-6, help='Finite difference epsilon')
    return parser.parse_args()


def load_expert_data(csv_path):
    """Load and extract expert demonstrations."""
    df = pd.read_csv(csv_path)
    
    # Extract columns
    t = df['t'].values
    q = np.column_stack([df['q1'].values, df['q2'].values])
    dq = np.column_stack([df['dq1'].values, df['dq2'].values])
    u = np.column_stack([df['u1'].values, df['u2'].values])
    
    return t, q, dq, u


def build_discrete_dynamics(dt, n=2, m=2):
    """
    Build discrete-time double integrator: x_{k+1} = A x_k + B u_k
    State: x = [q; dq]  (4D)
    Control: u = ddq    (2D)
    """
    # Continuous time: dx/dt = [0 I; 0 0] x + [0; I] u
    Ac = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    Bc = np.block([
        [np.zeros((n, m))],
        [np.eye(m)]
    ])
    
    # Discretize using matrix exponential
    sys_c = control.StateSpace(Ac, Bc, np.eye(4), np.zeros((4, 2)))
    sys_d = control.c2d(sys_c, dt, method='zoh')
    
    A = sys_d.A
    B = sys_d.B
    
    return A, B


def solve_discrete_riccati(A, B, Q, R):
    """Solve discrete-time algebraic Riccati equation."""
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return P, K
    except Exception as e:
        print(f"Warning: Riccati solver failed: {e}")
        return None, None


def compute_expert_gain(A, B, R, states, controls, ridge=1e-6):
    """
    Compute expert gain via least-squares: u ≈ -K x
    K = (X^T X + ridge*I)^{-1} X^T U
    """
    N = states.shape[0]
    X = states  # (N, 4)
    U = -controls  # (N, 2), note the negative sign
    
    # Regularized least squares
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    K_expert = linalg.solve(XtX, X.T @ U).T  # (2, 4)
    
    return K_expert


def project_Q_pd(Q, floor=1e-6):
    """Project Q to be positive definite."""
    Q_sym = 0.5 * (Q + Q.T)
    eigvals, eigvecs = linalg.eigh(Q_sym)
    eigvals = np.maximum(eigvals, floor)
    Q_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return Q_pd


def objective(Q, A, B, R, K_expert):
    """Compute objective function."""
    P, K = solve_discrete_riccati(A, B, Q, R)
    if K is None:
        return 1e10, None, None
    obj = np.sum((K - K_expert)**2)
    return obj, P, K


def compute_gradient_fd(Q, A, B, R, K_expert, eps=1e-6):
    """
    Compute gradient using finite differences.
    This is the CORRECT method - verified to work!
    """
    n = Q.shape[0]
    grad = np.zeros((n, n))
    
    # Get baseline objective
    obj_base, _, _ = objective(Q, A, B, R, K_expert)
    
    # Finite difference for each element
    for i in range(n):
        for j in range(n):
            Q_plus = Q.copy()
            Q_plus[i, j] += eps
            Q_plus = 0.5 * (Q_plus + Q_plus.T)  # Keep symmetric
            
            obj_plus, _, _ = objective(Q_plus, A, B, R, K_expert)
            grad[i, j] = (obj_plus - obj_base) / eps
    
    return grad


def ioc_batch_learn_Q(A, B, R, K_expert, Q_init, alpha=0.001, iters=2000, 
                      pd_floor=1e-6, grad_eps=1e-6, print_every=200):
    """
    Learn Q via gradient descent to match expert gain.
    Objective: min ||K_learned(Q) - K_expert||_F^2
    Uses FINITE DIFFERENCE gradients (verified correct).
    """
    Q = Q_init.copy()
    n = A.shape[0]
    
    print(f"\nStarting IOC batch learning:")
    print(f"  K_expert shape: {K_expert.shape}")
    print(f"  Q_init:\n{Q_init}")
    print(f"  Learning rate: {alpha}")
    print(f"  Max iterations: {iters}")
    print(f"  Gradient method: FINITE DIFFERENCES (eps={grad_eps})")
    
    for it in range(iters):
        # Project Q to be PD
        Q = project_Q_pd(Q, floor=pd_floor)
        
        # Compute objective
        obj, P, K = objective(Q, A, B, R, K_expert)
        
        if P is None or K is None:
            print(f"Iteration {it}: Riccati failed, stopping")
            break
        
        K_diff = K - K_expert
        
        if it % print_every == 0:
            print(f"Iter {it:4d}: obj={obj:.6f}, ||K_diff||_F={np.linalg.norm(K_diff, 'fro'):.6f}")
        
        # Compute gradient using finite differences
        grad_Q = compute_gradient_fd(Q, A, B, R, K_expert, eps=grad_eps)
        
        # Gradient descent update
        Q = Q - alpha * grad_Q
        
        # Early stopping
        if obj < 1e-8:
            print(f"Converged at iteration {it}")
            break
    
    # Final projection and solve
    Q = project_Q_pd(Q, floor=pd_floor)
    P, K = solve_discrete_riccati(A, B, Q, R)
    
    print(f"\nFinal results:")
    print(f"  Q_learned:\n{Q}")
    print(f"  K_learned:\n{K}")
    print(f"  K_expert:\n{K_expert}")
    print(f"  Final objective: {np.sum((K - K_expert)**2):.8f}")
    
    return Q, P, K


def main():
    args = parse_args()
    
    print("="*60)
    print("Discrete-Time Inverse Optimal Control")
    print("="*60)
    
    # Load expert data
    print(f"\nLoading expert data from: {args.csv}")
    t, q, dq, u = load_expert_data(args.csv)
    N = len(t)
    print(f"  Loaded {N} samples")
    
    # Build discrete dynamics
    dt = 1.0 / args.rate
    print(f"\nBuilding discrete dynamics (dt={dt:.4f} s)")
    A, B = build_discrete_dynamics(dt)
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    
    # Set up cost matrices
    R = np.eye(2)  # Control cost
    print(f"\nR matrix:\n{R}")
    
    # Compute state deviations
    q_des = np.array(args.q_des)
    dq_des = np.array(args.dq_des)
    print(f"\nDesired state: q_des={q_des}, dq_des={dq_des}")
    
    q_err = q - q_des
    dq_err = dq - dq_des
    states = np.column_stack([q_err, dq_err])  # (N, 4)
    controls = u  # (N, 2)
    
    print(f"  State deviations shape: {states.shape}")
    print(f"  Controls shape: {controls.shape}")
    
    # Extract expert gain
    print(f"\nExtracting expert gain K via least-squares...")
    K_expert = compute_expert_gain(A, B, R, states, controls, ridge=args.ridge)
    print(f"  K_expert:\n{K_expert}")
    
    # Check stability
    A_cl_expert = A - B @ K_expert
    eigs_expert = linalg.eigvals(A_cl_expert)
    print(f"  Closed-loop eigenvalues: {eigs_expert}")
    print(f"  Max |eig|: {np.max(np.abs(eigs_expert)):.4f} (stable if < 1)")
    
    # Initialize Q
    Q_init = np.eye(4)
    print(f"\nInitializing Q to identity")
    
    # Run IOC learning
    Q_learned, P_learned, K_learned = ioc_batch_learn_Q(
        A, B, R, K_expert, Q_init,
        alpha=args.alpha,
        iters=args.iters,
        pd_floor=args.pd_floor,
        grad_eps=args.grad_eps,
        print_every=args.print_every
    )
    
    # Save results
    print(f"\nSaving results to: {args.out}")
    np.savez(
        args.out,
        Q_learned=Q_learned,
        R=R,
        P_learned=P_learned,
        K_learned=K_learned,
        K_expert=K_expert,
        A=A,
        B=B,
        dt=dt,
        q_des=q_des,
        dq_des=dq_des
    )
    print("✓ Done!")


if __name__ == '__main__':
    main()
