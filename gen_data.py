#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from pathlib import Path

# ----------------------------
# Degradation injections
# ----------------------------

def inj_stuck_plane(x, plane, val=1.0):
    """Force one plane to a constant value."""
    y = x.copy()
    y[plane, :, :] = val
    return y

def inj_bias(x, plane, p=0.6):
    """Bias one plane to have higher probability of 1s."""
    y = x.copy()
    mask = (np.random.rand(*y[plane].shape) < p).astype(np.float32)
    y[plane] = mask
    return y

def inj_temporal(prev_x, rho=0.95):
    """
    Temporal correlation: new sample is mostly prev_x, with sparse flips.
    rho: higher => more correlated (fewer flips)
    """
    flips = (np.random.rand(*prev_x.shape) > rho).astype(np.float32)
    y = np.logical_xor(prev_x > 0.5, flips > 0.5).astype(np.float32)
    return y

# ----------------------------
# Healthy sample generators
# ----------------------------

def gen_healthy_sample(P, H, W, base_p=0.5):
    """
    Generate a "healthy" bit-plane stack.
    You can replace this with your real TRNG frame loader later.
    """
    x = (np.random.rand(P, H, W) < base_p).astype(np.float32)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output .npz path")
    ap.add_argument("--planes", type=int, default=6, help="Number of bit planes P")
    ap.add_argument("--H", type=int, default=64, help="Height")
    ap.add_argument("--W", type=int, default=64, help="Width")

    ap.add_argument("--n_healthy", type=int, default=2000)
    ap.add_argument("--n_degraded", type=int, default=2000)

    ap.add_argument("--temporal_rho", type=float, default=0.95)
    ap.add_argument("--bias_p", type=float, default=0.7)
    ap.add_argument("--stuck_val", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    P, H, W = args.planes, args.H, args.W

    X = []
    y = []
    metas = []

    # Healthy samples
    prev_for_temporal = None
    for i in range(args.n_healthy):
        x = gen_healthy_sample(P, H, W, base_p=0.5)
        X.append(x)
        y.append(0)
        metas.append({
            "label": 0,
            "inj_type": "healthy",
            "plane": None,
            "idx": i
        })
        prev_for_temporal = x if prev_for_temporal is None else prev_for_temporal

    # Degraded samples
    for j in range(args.n_degraded):
        # start from healthy
        x0 = gen_healthy_sample(P, H, W, base_p=0.5)
        inj = np.random.choice(["stuck-plane", "bias", "temporal-corr"])
        plane = int(np.random.randint(0, P))

        if inj == "stuck-plane":
            x = inj_stuck_plane(x0, plane=plane, val=args.stuck_val)
        elif inj == "bias":
            x = inj_bias(x0, plane=plane, p=args.bias_p)
        else:
            # temporal correlation uses previous sample (use last generated degraded/healthy)
            if prev_for_temporal is None:
                prev_for_temporal = x0
            x = inj_temporal(prev_for_temporal, rho=args.temporal_rho)
            prev_for_temporal = x

        X.append(x.astype(np.float32))
        y.append(1)
        metas.append({
            "label": 1,
            "inj_type": inj,
            "plane": plane,
            "idx": args.n_healthy + j
        })

    X = np.stack(X, axis=0)  # (N, P, H, W)
    y = np.array(y, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    metas = [metas[i] for i in perm.tolist()]

    # Save metas as JSON strings for portability
    metas_json = np.array([json.dumps(m) for m in metas], dtype=object)

    np.savez_compressed(str(out_path), x=X, y=y, meta=metas_json)
    print(f"[OK] Saved: {out_path}")
    print(f"     X shape: {X.shape}, y: {y.shape}, meta: {metas_json.shape}")

if __name__ == "__main__":
    main()
