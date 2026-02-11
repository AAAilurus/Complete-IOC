#!/usr/bin/env python3
"""
Offline IOC (UPDATED): Learn DIAGONAL Q from leader's K* matrix

Goal:
  Minimize:  J(q) = ||K(Q(q)) - K*||_F^2
  where Q(q) = diag(q0,q1,q2,q3), R fixed.

Why this update:
  - Identifiable + interpretable intent (position/velocity weights)
  - Avoids off-diagonal ambiguity
  - FD gradient is correct and much faster (4 params instead of 16)
"""

import argparse
import numpy as np
from scipy import linalg
import control
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ CLI ------------------------ #
def parse_args():
    p = argparse.ArgumentParser(description="Offline IOC: learn diagonal Q from K*")
    p.add_argument("--k_star", type=float, nargs="+", required=True,
                   help="K* as flat array (8 values for 2x4)")
    p.add_argument("--out", default="/root/so100_ws/ioc_results/learned_Q.npz",
                   help="Output .npz file")
    p.add_argument("--rate", type=float, default=50.0, help="Control rate (Hz)")
    p.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
    p.add_argument("--iters", type=int, default=3000, help="Max iterations")
    p.add_argument("--print_every", type=int, default=200, help="Print interval")
    p.add_argument("--history_csv", default="/root/so100_ws/ioc_results/ioc_history.csv",
                   help="History CSV file")
    p.add_argument("--plot", default="/root/so100_ws/ioc_results/ioc_convergence.png",
                   help="Convergence plot file")
    p.add_argument("--eps_fd", type=float, default=1e-6,
                   help="Finite-difference step for diagonal entries")
    p.add_argument("--q_init", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                   help="Initial diagonal Q entries (4 numbers)")
    p.add_argument("--q_floor", type=float, default=1e-6,
                   help="Minimum eigenvalue floor (keeps Q positive definite)")
    p.add_argument("--tol_k", type=float, default=1e-5,
                   help="Stop when ||K-K*||_F < tol_k")
    return p.parse_args()


# ------------------------ System model ------------------------ #
def build_discrete_dynamics(dt, n=2, m=2):
    """Discrete-time double integrator with ZOH: x=[q;dq], u=ddq."""
    Ac = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    Bc = np.block([
        [np.zeros((n, m))],
        [np.eye(m)]
    ])
    sys_c = control.StateSpace(Ac, Bc, np.eye(2*n), np.zeros((2*n, m)))
    sys_d = control.c2d(sys_c, dt, method="zoh")
    return sys_d.A, sys_d.B


def solve_riccati(A, B, Q, R):
    """Return (P,K) for discrete LQR, or (None,None) if failure."""
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        S = R + B.T @ P @ B
        K = linalg.solve(S, (B.T @ P @ A))  # more stable than inv()
        return P, K
    except Exception:
        return None, None


# ------------------------ Q handling ------------------------ #
def diag_to_Q(q_diag):
    """q_diag (4,) -> Q (4x4) diagonal matrix."""
    return np.diag(q_diag)


def project_diag_floor(q_diag, floor=1e-6):
    """
    Since Q is diagonal, PD projection is just flooring diagonals.
    (For general Q you'd do eigen projection; for diagonal this is enough.)
    """
    q = np.array(q_diag, dtype=float).copy()
    q[q < floor] = floor
    return q


# ------------------------ Objective + Gradient ------------------------ #
def objective_from_qdiag(q_diag, A, B, R, K_star):
    """Compute objective J and K for current diagonal q_diag."""
    Q = diag_to_Q(q_diag)
    P, K = solve_riccati(A, B, Q, R)
    if K is None:
        return 1e12, None, None
    E = K - K_star
    J = float(np.sum(E * E))  # ||K-K*||_F^2
    return J, K, P


def grad_fd_diag(q_diag, A, B, R, K_star, eps=1e-6):
    """
    Finite-difference gradient wrt ONLY the 4 diagonal entries.
    Returns grad_q (4,).
    """
    q = np.array(q_diag, dtype=float)
    J0, _, _ = objective_from_qdiag(q, A, B, R, K_star)
    g = np.zeros_like(q)

    for i in range(len(q)):
        qp = q.copy()
        qp[i] += eps
        Jp, _, _ = objective_from_qdiag(qp, A, B, R, K_star)
        g[i] = (Jp - J0) / eps

    return g


# ------------------------ Learning loop ------------------------ #
def learn_qdiag(A, B, R, K_star, alpha, iters, print_every, eps_fd, q_init, q_floor, tol_k):
    q_diag = project_diag_floor(q_init, q_floor)

    history = {
        "iteration": [],
        "objective": [],
        "k_error": [],
        "q00": [], "q11": [], "q22": [], "q33": []
    }

    print("\n================ Offline IOC (Diagonal Q) ================")
    print("Target K*:\n", K_star)
    print(f"alpha={alpha}, iters={iters}, eps_fd={eps_fd}, q_floor={q_floor}, tol_k={tol_k}")
    print("Initial q_diag =", q_diag.tolist())
    print("=========================================================\n")

    best = {"J": np.inf, "q": q_diag.copy(), "K": None, "P": None}

    for it in range(iters):
        # ensure PD
        q_diag = project_diag_floor(q_diag, q_floor)

        J, K, P = objective_from_qdiag(q_diag, A, B, R, K_star)
        if K is None:
            # rare, but keep safe
            J = 1e12
            k_err = np.inf
        else:
            k_err = float(np.linalg.norm(K - K_star, ord="fro"))

        # record
        history["iteration"].append(it)
        history["objective"].append(J)
        history["k_error"].append(k_err)
        history["q00"].append(q_diag[0])
        history["q11"].append(q_diag[1])
        history["q22"].append(q_diag[2])
        history["q33"].append(q_diag[3])

        # keep best
        if J < best["J"] and K is not None:
            best = {"J": J, "q": q_diag.copy(), "K": K.copy(), "P": P.copy()}

        if it % print_every == 0:
            print(f"Iter {it:4d}: J={J:.10e}, ||K-K*||_F={k_err:.10e}, q={q_diag.tolist()}")

        # stopping criterion based on K match (what you actually care about)
        if K is not None and k_err < tol_k:
            print(f"\n✓ Converged at iter {it}: ||K-K*||_F={k_err:.3e}")
            break

        # gradient step
        g = grad_fd_diag(q_diag, A, B, R, K_star, eps=eps_fd)
        q_diag = q_diag - alpha * g

    # return best (safer than last)
    q_best = project_diag_floor(best["q"], q_floor)
    Q_best = diag_to_Q(q_best)
    P_best, K_best = solve_riccati(A, B, Q_best, R)

    # compute final stats
    E = K_best - K_star
    J_final = float(np.sum(E * E))
    k_err_final = float(np.linalg.norm(E, ord="fro"))

    print("\n==================== Final (Best) ====================")
    print("Learned q_diag =", q_best.tolist())
    print("Q_learned =\n", Q_best)
    print("K_learned =\n", K_best)
    print("Final J      =", f"{J_final:.12e}")
    print("Final ||K-K*||_F =", f"{k_err_final:.12e}")
    print("======================================================\n")

    return Q_best, P_best, K_best, history


# ------------------------ Plotting ------------------------ #
def plot_convergence(history, plot_file):
    it = np.array(history["iteration"])
    kerr = np.array(history["k_error"])
    obj = np.array(history["objective"])

    q00 = np.array(history["q00"])
    q11 = np.array(history["q11"])
    q22 = np.array(history["q22"])
    q33 = np.array(history["q33"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(it, kerr, linewidth=2)
    axes[0, 0].set_title("||K - K*||_F")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("K error")
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(it, obj, linewidth=2)
    axes[0, 1].set_title("Objective J = ||K-K*||_F^2")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("J")
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(it, q00, label="Q[0,0]", linewidth=2)
    axes[1, 0].plot(it, q11, label="Q[1,1]", linewidth=2)
    axes[1, 0].set_title("Position weights")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(it, q22, label="Q[2,2]", linewidth=2)
    axes[1, 1].plot(it, q33, label="Q[3,3]", linewidth=2)
    axes[1, 1].set_title("Velocity weights")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to: {plot_file}")


# ------------------------ Main ------------------------ #
def main():
    args = parse_args()

    print("=" * 60)
    print("Offline IOC (UPDATED): Learn Diagonal Q from K*")
    print("=" * 60)

    if len(args.k_star) != 8:
        raise ValueError("K* must have exactly 8 values (2x4).")

    K_star = np.array(args.k_star, dtype=float).reshape(2, 4)

    dt = 1.0 / args.rate
    A, B = build_discrete_dynamics(dt)
    R = np.eye(2)

    print("\nSystem:")
    print(f"  dt = {dt:.6f}")
    print(f"  A shape = {A.shape}, B shape = {B.shape}")
    print("  R =\n", R)

    # Learn
    Q, P, K, history = learn_qdiag(
        A=A, B=B, R=R, K_star=K_star,
        alpha=args.alpha, iters=args.iters, print_every=args.print_every,
        eps_fd=args.eps_fd, q_init=np.array(args.q_init, dtype=float),
        q_floor=args.q_floor, tol_k=args.tol_k
    )

    # Save history CSV
    if args.history_csv:
        df = pd.DataFrame(history)
        df.to_csv(args.history_csv, index=False)
        print(f"✓ History saved to: {args.history_csv}")

    # Plot
    if args.plot:
        plot_convergence(history, args.plot)

    # Save learned matrices
    print(f"\nSaving learned result to: {args.out}")
    np.savez(args.out, Q=Q, P=P, K=K, K_star=K_star, A=A, B=B, R=R, dt=dt)
    print("✓ Done!")


if __name__ == "__main__":
    main()
