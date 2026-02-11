#!/usr/bin/env python3
"""
Offline IOC: Learn Q matrix from leader's K* matrix

Goal:
    minimize  J(Q) = || K(Q) - K* ||_F^2

Where:
    K(Q) is the discrete-time LQR gain produced by (A,B,Q,R)

We update Q by gradient descent using finite-difference gradients.
We keep Q symmetric positive definite by eigenvalue projection.

Outputs:
    - learned_Q.npz   (Q,P,K,K*,A,B,R,dt)
    - ioc_history.csv (objective, k_error, diag(Q))
    - ioc_convergence.png
"""

import argparse
import numpy as np
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Offline IOC: learn Q from K*')
    p.add_argument('--k_star', type=float, nargs='+', required=True,
                   help='K* as flat array (8 values for 2x4)')
    p.add_argument('--out', default='/root/so100_ws/ioc_results/learned_Q.npz')
    p.add_argument('--rate', type=float, default=50.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--iters', type=int, default=3000)
    p.add_argument('--print_every', type=int, default=200)
    p.add_argument('--eps_fd', type=float, default=1e-6, help='finite diff step')
    p.add_argument('--pd_floor', type=float, default=1e-6, help='min eigenvalue of Q')
    p.add_argument('--history_csv', default='/root/so100_ws/ioc_results/ioc_history.csv')
    p.add_argument('--plot', default='/root/so100_ws/ioc_results/ioc_convergence.png')
    return p.parse_args()


# ----------------------------
# Dynamics (same as leader)
# ----------------------------
def build_discrete_dynamics(dt: float, n: int = 2):
    I = np.eye(n)
    Z = np.zeros((n, n))
    A = np.block([[I, dt * I],
                  [Z, I]])
    B = np.block([[0.5 * (dt ** 2) * I],
                  [dt * I]])
    return A, B


# ----------------------------
# LQR solve: DARE -> K
# ----------------------------
def solve_dlqr(A, B, Q, R):
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, (B.T @ P @ A))
        return P, K
    except Exception:
        return None, None


# ----------------------------
# SPD projection for Q
# ----------------------------
def project_Q_pd(Q, floor=1e-6):
    Qs = 0.5 * (Q + Q.T)
    eigvals, eigvecs = linalg.eigh(Qs)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ----------------------------
# Objective J(Q) and K(Q)
# ----------------------------
def objective(Q, A, B, R, K_star):
    P, K = solve_dlqr(A, B, Q, R)
    if K is None:
        return 1e12, None
    E = K - K_star
    J = float(np.sum(E * E))  # Frobenius^2
    return J, K


# ----------------------------
# Finite-difference gradient dJ/dQ
# (Full matrix; you can restrict to diag if you want.)
# ----------------------------
def compute_gradient_fd(Q, A, B, R, K_star, eps=1e-6):
    n = Q.shape[0]
    grad = np.zeros((n, n), dtype=float)

    J0, _ = objective(Q, A, B, R, K_star)

    # Symmetric perturbations are more consistent:
    # Q_plus = Q + eps*(E_ij + E_ji)/2 for i!=j, and Q+eps*E_ii for i==j
    for i in range(n):
        for j in range(n):
            Qp = Q.copy()
            if i == j:
                Qp[i, i] += eps
            else:
                Qp[i, j] += 0.5 * eps
                Qp[j, i] += 0.5 * eps

            Jp, _ = objective(Qp, A, B, R, K_star)
            grad[i, j] = (Jp - J0) / eps

    # Keep grad symmetric (important)
    grad = 0.5 * (grad + grad.T)
    return grad


# ----------------------------
# Main learning loop
# ----------------------------
def learn_Q(A, B, R, K_star, alpha, iters, print_every, eps_fd, pd_floor):
    n = A.shape[0]

    # Initialize Q
    Q = np.eye(n)

    history = {
        'iter': [],
        'J': [],
        'k_error': [],
        'q00': [], 'q11': [], 'q22': [], 'q33': []
    }

    for it in range(iters):
        Q = project_Q_pd(Q, pd_floor)

        J, K = objective(Q, A, B, R, K_star)
        k_err = float(np.linalg.norm(K - K_star, ord='fro'))

        history['iter'].append(it)
        history['J'].append(J)
        history['k_error'].append(k_err)
        history['q00'].append(Q[0, 0])
        history['q11'].append(Q[1, 1])
        history['q22'].append(Q[2, 2])
        history['q33'].append(Q[3, 3])

        if it % print_every == 0:
            print(f"Iter {it:4d}: J={J:.6e}, ||K-K*||_F={k_err:.6e}, diag(Q)={np.diag(Q)}")

        # Stop condition (you can tune)
        if k_err < 1e-5:
            print(f"✓ Converged at iter {it}: ||K-K*||_F={k_err:.6e}")
            break

        grad = compute_gradient_fd(Q, A, B, R, K_star, eps=eps_fd)

        # Gradient descent update
        Q = Q - alpha * grad

    Q = project_Q_pd(Q, pd_floor)
    P, K = solve_dlqr(A, B, Q, R)
    return Q, P, K, history


def plot_convergence(history, plot_file):
    it = np.array(history['iter'])
    kerr = np.array(history['k_error'])
    J = np.array(history['J'])

    q00 = np.array(history['q00'])
    q11 = np.array(history['q11'])
    q22 = np.array(history['q22'])
    q33 = np.array(history['q33'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(it, kerr, linewidth=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    axes[0, 0].set_title('||K - K*||_F')
    axes[0, 0].set_xlabel('Iteration')

    axes[0, 1].plot(it, J, linewidth=2)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Objective J(Q)')
    axes[0, 1].set_xlabel('Iteration')

    axes[1, 0].plot(it, q00, label='Q[0,0]', linewidth=2)
    axes[1, 0].plot(it, q11, label='Q[1,1]', linewidth=2)
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Position weights')
    axes[1, 0].legend()

    axes[1, 1].plot(it, q22, label='Q[2,2]', linewidth=2)
    axes[1, 1].plot(it, q33, label='Q[3,3]', linewidth=2)
    axes[1, 1].grid(True)
    axes[1, 1].set_title('Velocity weights')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_file}")


def main():
    args = parse_args()

    if len(args.k_star) != 8:
        raise ValueError("K* must have 8 values (2x4).")

    K_star = np.array(args.k_star, dtype=float).reshape(2, 4)

    dt = 1.0 / args.rate
    A, B = build_discrete_dynamics(dt, n=2)
    R = np.eye(2)

    print("=" * 70)
    print("Offline IOC: Learn Q from K*")
    print("=" * 70)
    print(f"dt={dt:.4f}, A.shape={A.shape}, B.shape={B.shape}")
    print(f"K*=\n{K_star}\n")

    Q, P, K, history = learn_Q(
        A=A, B=B, R=R, K_star=K_star,
        alpha=args.alpha, iters=args.iters,
        print_every=args.print_every,
        eps_fd=args.eps_fd, pd_floor=args.pd_floor
    )

    print("\nFinal:")
    print(f"Q=\n{Q}")
    print(f"K=\n{K}")
    print(f"||K-K*||_F = {np.linalg.norm(K-K_star, 'fro'):.10e}")

    # Save history
    df = pd.DataFrame(history)
    df.to_csv(args.history_csv, index=False)
    print(f"✓ History saved: {args.history_csv}")

    # Plot
    plot_convergence(history, args.plot)

    # Save learned matrices
    np.savez(args.out, Q=Q, P=P, K=K, K_star=K_star, A=A, B=B, R=R, dt=dt)
    print(f"✓ Saved learned_Q: {args.out}")


if __name__ == '__main__':
    main()
