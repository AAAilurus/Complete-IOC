#!/usr/bin/env python3
"""
Offline IOC: Learn Q matrix from leader's K* matrix
Minimizes: ||K_learned(Q) - K*||_F^2
Records convergence history
"""
import argparse
import numpy as np
from scipy import linalg
import control
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Learn Q from K*')
    parser.add_argument('--k_star', type=float, nargs='+', required=True, 
                        help='K* matrix as flat array (8 values for 2x4)')
    parser.add_argument('--out', default='/root/so100_ws/ioc_results/learned_Q.npz', help='Output file')
    parser.add_argument('--rate', type=float, default=50.0, help='Control rate (Hz)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--iters', type=int, default=3000, help='Max iterations')
    parser.add_argument('--print_every', type=int, default=200, help='Print interval')
    parser.add_argument('--history_csv', default='/root/so100_ws/ioc_results/ioc_history.csv', help='Convergence history CSV')
    parser.add_argument('--plot', default='/root/so100_ws/ioc_results/ioc_convergence.png', help='Convergence plot')
    return parser.parse_args()


def build_discrete_dynamics(dt, n=2, m=2):
    """Build discrete-time double integrator."""
    Ac = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    Bc = np.block([
        [np.zeros((n, m))],
        [np.eye(m)]
    ])
    
    sys_c = control.StateSpace(Ac, Bc, np.eye(4), np.zeros((4, 2)))
    sys_d = control.c2d(sys_c, dt, method='zoh')
    
    return sys_d.A, sys_d.B


def solve_riccati(A, B, Q, R):
    """Solve discrete-time Riccati."""
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return P, K
    except:
        return None, None


def project_Q_pd(Q, floor=1e-6):
    """Project Q to positive definite."""
    Q_sym = 0.5 * (Q + Q.T)
    eigvals, eigvecs = linalg.eigh(Q_sym)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def objective(Q, A, B, R, K_star):
    """Compute objective."""
    P, K = solve_riccati(A, B, Q, R)
    if K is None:
        return 1e10, None
    return np.sum((K - K_star)**2), K


def compute_gradient_fd(Q, A, B, R, K_star, eps=1e-6):
    """Finite difference gradient."""
    n = Q.shape[0]
    grad = np.zeros((n, n))
    obj_base, _ = objective(Q, A, B, R, K_star)
    
    for i in range(n):
        for j in range(n):
            Q_plus = Q.copy()
            Q_plus[i, j] += eps
            Q_plus = 0.5 * (Q_plus + Q_plus.T)
            obj_plus, _ = objective(Q_plus, A, B, R, K_star)
            grad[i, j] = (obj_plus - obj_base) / eps
    
    return grad


def learn_Q(A, B, R, K_star, alpha=0.5, iters=3000, print_every=200, history_csv=None):
    """Learn Q via gradient descent and record history."""
    Q = np.eye(4)  # Start from identity
    
    # History tracking
    history = {
        'iteration': [],
        'objective': [],
        'k_error': [],
        'q00': [], 'q11': [], 'q22': [], 'q33': []
    }
    
    print(f"\nStarting IOC Learning:")
    print(f"  K* (target):\n{K_star}")
    print(f"  Learning rate: {alpha}")
    print(f"  Max iterations: {iters}\n")
    
    for it in range(iters):
        Q = project_Q_pd(Q)
        
        obj, K = objective(Q, A, B, R, K_star)
        k_error = np.linalg.norm(K - K_star, 'fro')
        
        # Record history
        history['iteration'].append(it)
        history['objective'].append(obj)
        history['k_error'].append(k_error)
        history['q00'].append(Q[0,0])
        history['q11'].append(Q[1,1])
        history['q22'].append(Q[2,2])
        history['q33'].append(Q[3,3])
        
        if it % print_every == 0:
            print(f"Iter {it:4d}: obj={obj:.8f}, ||K-K*||_F={k_error:.8f}")
        
        grad = compute_gradient_fd(Q, A, B, R, K_star)
        Q = Q - alpha * grad
        
        if obj < 1e-10:
            print(f"Converged at iteration {it}")
            break
    
    Q = project_Q_pd(Q)
    P, K = solve_riccati(A, B, Q, R)
    
    print(f"\nFinal Results:")
    print(f"  Q_learned:\n{Q}")
    print(f"  K_learned:\n{K}")
    print(f"  K* (target):\n{K_star}")
    print(f"  Final objective: {obj:.10f}")
    print(f"  Final error: {k_error:.10f}")
    
    # Save history
    if history_csv:
        df = pd.DataFrame(history)
        df.to_csv(history_csv, index=False)
        print(f"\n✓ History saved to: {history_csv}")
    
    return Q, P, K, history


def plot_convergence(history, plot_file):
    """Plot convergence curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # K error over time
    axes[0, 0].plot(history['iteration'], history['k_error'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('||K - K*||_F')
    axes[0, 0].set_title('Gain Matrix Error Convergence')
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Objective over time
    axes[0, 1].plot(history['iteration'], history['objective'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Objective')
    axes[0, 1].set_title('Objective Function')
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # Q diagonal elements
    axes[1, 0].plot(history['iteration'], history['q00'], label='Q[0,0] (target=10)', linewidth=2)
    axes[1, 0].plot(history['iteration'], history['q11'], label='Q[1,1] (target=10)', linewidth=2)
    axes[1, 0].axhline(y=10, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Q value')
    axes[1, 0].set_title('Q Position Weights')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['iteration'], history['q22'], label='Q[2,2] (target=1)', linewidth=2)
    axes[1, 1].plot(history['iteration'], history['q33'], label='Q[3,3] (target=1)', linewidth=2)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Q value')
    axes[1, 1].set_title('Q Velocity Weights')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    plt.close()


def main():
    args = parse_args()
    
    print("="*60)
    print("Offline IOC: Learning Q from K*")
    print("="*60)
    
    # Parse K*
    K_star = np.array(args.k_star).reshape(2, 4)
    
    # Build dynamics
    dt = 1.0 / args.rate
    A, B = build_discrete_dynamics(dt)
    R = np.eye(2)
    
    print(f"\nSystem Info:")
    print(f"  dt = {dt:.4f} s")
    print(f"  A shape: {A.shape}")
    print(f"  B shape: {B.shape}")
    print(f"  R:\n{R}")
    
    # Learn Q
    Q, P, K, history = learn_Q(A, B, R, K_star, alpha=args.alpha, iters=args.iters, 
                                print_every=args.print_every, history_csv=args.history_csv)
    
    # Plot convergence
    plot_convergence(history, args.plot)
    
    # Save
    print(f"\nSaving to: {args.out}")
    np.savez(args.out, Q=Q, P=P, K=K, K_star=K_star, A=A, B=B, R=R, dt=dt)
    print("✓ Done!")


if __name__ == '__main__':
    main()
