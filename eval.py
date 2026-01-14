#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def topk_mean(residual_flat, frac=0.01):
    B, D = residual_flat.shape
    k = max(1, int(D * frac))
    topk = np.partition(residual_flat, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)

def plane_imbalance(plane_mse):
    return plane_mse.max(axis=1) / (plane_mse.mean(axis=1) + 1e-8)

def window_max_median(scores_1d, win=16):
    out = np.zeros_like(scores_1d, dtype=np.float32)
    for i in range(len(scores_1d)):
        s = scores_1d[max(0, i - win + 1): i + 1]
        out[i] = float(np.max(s) - np.median(s))
    return out

def metrics_roc_pr(scores, labels, fpr_points=(0.01, 0.001)):
    # labels must be binary 0/1
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(labels, scores, pos_label=1)
    pr_auc = auc(rec, prec)

    out = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}
    for fp in fpr_points:
        idx = np.searchsorted(fpr, fp, side="left")
        idx = min(idx, len(tpr) - 1)
        out[f"tpr_at_fpr_{fp:g}"] = float(tpr[idx])
    return out

import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="outputs/eval_report.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--top_frac", type=float, default=0.01)
    ap.add_argument("--fpr_points", type=str, default="0.01,0.001")
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fpr_points = tuple(float(x.strip()) for x in args.fpr_points.split(",") if x.strip())

    d = np.load(args.npz, allow_pickle=True)
    X = d["x"].astype(np.float32)  # (N,P,H,W)
    y = d["y"].astype(np.int64)
    meta_json = d["meta"]

    ck = torch.load(args.ckpt, map_location="cpu")
    in_ch = int(ck.get("in_ch", X.shape[1]))
    model = ConvAE(in_ch=in_ch)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    N = X.shape[0]
    scores_tail = np.zeros((N,), dtype=np.float32)
    scores_plane = np.zeros((N,), dtype=np.float32)
    scores_frame = np.zeros((N,), dtype=np.float32)

    idx = 0
    while idx < N:
        j = min(N, idx + args.batch)
        xb = torch.from_numpy(X[idx:j]).to(device)
        xhat = model(xb)

        # squared error (MSE-style)
        res = (xb - xhat)
        res = (res * res).detach().cpu().numpy()  # (B,P,H,W)

        plane_mse = res.mean(axis=(2, 3))          # (B,P)
        s_frame = plane_mse.mean(axis=1)           # (B,)
        s_plane = plane_imbalance(plane_mse)       # (B,)

        flat = res.reshape(res.shape[0], -1)
        s_tail = topk_mean(flat, frac=args.top_frac)

        scores_tail[idx:j] = s_tail.astype(np.float32)
        scores_plane[idx:j] = s_plane.astype(np.float32)
        scores_frame[idx:j] = s_frame.astype(np.float32)
        idx = j

    scores_temp = window_max_median(scores_tail, win=args.win)

    # IMPORTANT: binarize labels: 0=healthy, 1=any anomaly
    labels = (y != 0).astype(np.int64)

    report = {
        "config": {
            "npz": str(args.npz),
            "ckpt": str(args.ckpt),
            "batch": args.batch,
            "win": args.win,
            "top_frac": args.top_frac,
            "fpr_points": list(fpr_points),
        },
        "metrics": {
            "tail": metrics_roc_pr(scores_tail, labels, fpr_points=fpr_points),
            "plane": metrics_roc_pr(scores_plane, labels, fpr_points=fpr_points),
            "temporal": metrics_roc_pr(scores_temp, labels, fpr_points=fpr_points),
            "frame_baseline": metrics_roc_pr(scores_frame, labels, fpr_points=fpr_points),
        },
        "scores_summary": {
            "tail": {"mean": float(scores_tail.mean()), "std": float(scores_tail.std())},
            "plane": {"mean": float(scores_plane.mean()), "std": float(scores_plane.std())},
            "temporal": {"mean": float(scores_temp.mean()), "std": float(scores_temp.std())},
            "frame": {"mean": float(scores_frame.mean()), "std": float(scores_frame.std())},
        }
    }

    # Per-injection-type breakdown (optional)
    metas = [json.loads(s) for s in meta_json.tolist()]
    inj_types = sorted(list(set(m.get("inj_type", "unknown") for m in metas)))
    report["per_inj_type"] = {}

    for t in inj_types:
        mask = np.array([m.get("inj_type") == t for m in metas], dtype=bool)
        if mask.sum() < 5:
            continue
        lbl = labels[mask]
        if len(np.unique(lbl)) < 2:
            continue
        report["per_inj_type"][t] = {
            "tail": metrics_roc_pr(scores_tail[mask], lbl, fpr_points=fpr_points),
            "plane": metrics_roc_pr(scores_plane[mask], lbl, fpr_points=fpr_points),
            "temporal": metrics_roc_pr(scores_temp[mask], lbl, fpr_points=fpr_points),
            "frame_baseline": metrics_roc_pr(scores_frame[mask], lbl, fpr_points=fpr_points),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved report: {out_json}")
    print(json.dumps(report["metrics"], indent=2))

if __name__ == "__main__":
    main()
