#!/usr/bin/env python3
import numpy as np
import argparse, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='/tmp/ioc_result.npz')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    X, U = [], []
    with open(args.csv, 'r') as f:
        r = csv.DictReader(f)
        need = ['q1','q2','dq1','dq2','u1','u2']
        for k in need:
            if k not in r.fieldnames:
                raise RuntimeError(f"CSV missing '{k}', got columns: {r.fieldnames}")
        for row in r:
            x = [float(row['q1']), float(row['q2']), float(row['dq1']), float(row['dq2'])]
            u = [float(row['u1']), float(row['u2'])]
            X.append(x); U.append(u)

    X = np.asarray(X, dtype=float)  # N x 4
    U = np.asarray(U, dtype=float)  # N x 2
    print(f"Loaded N={X.shape[0]} samples")

    if X.shape[0] < 20:
        raise RuntimeError("Too few samples. Increase duration or ensure joint_states are received.")

    # Solve U ≈ -X K^T  => least squares
    # lstsq gives A in X @ A ≈ U, then K = -A^T
    A, *_ = np.linalg.lstsq(X, U, rcond=None)   # A: 4x2
    K = -A.T                                     # 2x4

    print("K_hat (2x4):\n", K)
    np.savez(args.out, K=K, csv=args.csv, N=X.shape[0])
    print("Saved:", args.out)

if __name__ == '__main__':
    main()
