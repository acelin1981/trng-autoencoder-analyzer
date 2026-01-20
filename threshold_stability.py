#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold stability (B): Q99 / Q99.9 vs epoch (plateau evidence)

What it does:
- Trains the same ConvAE used in your paper on healthy-only data
- After EACH epoch, computes frame_baseline scores on val_healthy (same split file)
- Logs Q99 and Q99.9 thresholds vs epoch to JSON + prints a compact table

This produces direct evidence for "threshold plateau / stability".
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import eval as E  # uses the same ConvAE as in your paper code


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    # your repo uses either "x" or "X" depending on file
    if "x" in d.files:
        return d["x"].astype(np.float32)
    if "X" in d.files:
        return d["X"].astype(np.float32)
    raise KeyError(f"Unsupported keys in {path}: {d.files}")


@torch.no_grad()
def frame_baseline_scores(model: torch.nn.Module, X: np.ndarray, batch: int, device: torch.device):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        xhat = model(xb)
        res = (xb - xhat).abs()
        # frame_baseline = mean residual energy over (P,H,W)
        s = res.mean(dim=(1, 2, 3))
        out.append(s.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/train_healthy_6ch.npz")
    ap.add_argument("--val", type=str, default="data/val_healthy_6ch.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--denoise", action="store_true", help="Apply denoise augmentation during training")
    ap.add_argument("--noise-p", type=float, default=0.01, help="Bit flip probability when --denoise is enabled")
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--out", type=str, default="threshold_stability.json")
    ap.add_argument("--last-k", type=int, default=10, help="Also print summary for last K epochs")
    args = ap.parse_args()

    set_seed(args.seed)

    Xtr = load_npz(Path(args.train))
    Xv = load_npz(Path(args.val))

    in_ch = int(Xtr.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = E.ConvAE(in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    def iter_batches(X):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for i in range(0, len(X), args.batch):
            j = idx[i:i+args.batch]
            yield X[j]

    log = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb_np in iter_batches(Xtr):
            xb = torch.from_numpy(xb_np).to(device)

            if args.denoise:
                # simple bit-flip noise; keeps values in {0,1} if input is {0,1}
                flip = (torch.rand_like(xb) < args.noise_p).float()
                xb_in = (xb + flip) % 2.0
            else:
                xb_in = xb

            opt.zero_grad()
            xhat = model(xb_in)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # thresholds on healthy validation set
        s = frame_baseline_scores(model, Xv, batch=args.eval_batch, device=device)
        q99 = float(np.quantile(s, 0.99))
        q999 = float(np.quantile(s, 0.999))

        rec = {
            "epoch": ep,
            "train_mse": float(np.mean(losses)),
            "q99": q99,
            "q999": q999,
        }
        log.append(rec)
        print(f"[epoch {ep:02d}] train_mse={rec['train_mse']:.6f}  q99={q99:.6f}  q99.9={q999:.6f}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"config": vars(args), "log": log}, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved: {out_path}")

    # Print a compact last-K table (plateau evidence)
    k = max(1, min(args.last_k, len(log)))
    tail = log[-k:]
    q99_vals = np.array([r["q99"] for r in tail], dtype=np.float64)
    q999_vals = np.array([r["q999"] for r in tail], dtype=np.float64)

    def summ(x):
        return float(x.mean()), float(x.std(ddof=1)), float(x.min()), float(x.max())

    q99_m, q99_s, q99_min, q99_max = summ(q99_vals)
    q999_m, q999_s, q999_min, q999_max = summ(q999_vals)

    print(f"\n=== Threshold plateau summary (last {k} epochs) ===")
    print(f"Q99   : mean={q99_m:.6f}, std={q99_s:.6f}, range=[{q99_min:.6f}, {q99_max:.6f}]")
    print(f"Q99.9 : mean={q999_m:.6f}, std={q999_s:.6f}, range=[{q999_min:.6f}, {q999_max:.6f}]")

if __name__ == "__main__":
    main()
