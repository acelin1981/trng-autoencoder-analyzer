#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# -------------------------
# Model (must match train.py)
# -------------------------
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
def compute_frame_scores(npz_path: str, ckpt_path: str, batch: int = 64):
    """
    Returns:
      scores_frame: (N,) float32 (frame_baseline score = mean MSE over planes and pixels)
      labels: (N,) int64    (0=healthy, 1=anomaly)
    """
    d = np.load(npz_path, allow_pickle=True)

    # robust key loading
    def pick(names):
        for n in names:
            if n in d.files:
                return d[n]
        return None

    X = pick(["x", "X", "data", "arr_0"])
    y = pick(["y", "Y", "label", "labels", "arr_1"])

    if X is None or y is None:
        raise KeyError(f"Cannot find x/y in {npz_path}. keys={d.files}")

    X = X.astype(np.float32)  # (N,P,H,W)
    y = y.astype(np.int64)

    ck = torch.load(ckpt_path, map_location="cpu")
    in_ch = int(ck.get("in_ch", X.shape[1]))

    if X.shape[1] != in_ch:
        raise RuntimeError(
            f"Channel mismatch: data has {X.shape[1]} channels, ckpt expects {in_ch}. "
            f"Use a matching train/eval channel setup."
        )

    model = ConvAE(in_ch=in_ch)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    N = X.shape[0]
    scores_frame = np.zeros((N,), dtype=np.float32)

    idx = 0
    while idx < N:
        j = min(N, idx + batch)
        xb = torch.from_numpy(X[idx:j]).to(device)
        xhat = model(xb)

        # squared error (true MSE-style)
        res = xb - xhat
        res = (res * res).detach().cpu().numpy()  # (B,P,H,W)

        plane_mse = res.mean(axis=(2, 3))      # (B,P)
        s_frame = plane_mse.mean(axis=1)       # (B,)
        scores_frame[idx:j] = s_frame.astype(np.float32)
        idx = j

    labels = (y != 0).astype(np.int64)
    return scores_frame, labels


def plot_roc(scores, labels, out_png: Path, title: str):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xscale("log")
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}\nAUC = {roc_auc:.4f}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return float(roc_auc)


def plot_pr(scores, labels, out_png: Path, title: str):
    prec, rec, _ = precision_recall_curve(labels, scores, pos_label=1)
    pr_auc = auc(rec, prec)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title}\nPR-AUC = {pr_auc:.4f}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return float(pr_auc)


def plot_hist(scores, labels, out_png: Path, title: str, bins: int = 60):
    s0 = scores[labels == 0]
    s1 = scores[labels == 1]

    plt.figure()
    plt.hist(s0, bins=bins, alpha=0.6, density=True, label="Healthy (label=0)")
    plt.hist(s1, bins=bins, alpha=0.6, density=True, label="Anomaly (label=1)")
    plt.xlabel("frame_baseline score (mean MSE)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_auc_compare(auc_base, auc_dn, out_png: Path, title: str):
    plt.figure()
    plt.bar(["Baseline", "Denoise"], [auc_base, auc_dn])
    plt.ylim(0.0, 1.0)
    plt.ylabel("ROC-AUC (frame_baseline)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="e.g., data/eval_mix.npz")
    ap.add_argument("--ckpt", required=True, help="denoise ckpt path, e.g., checkpoints_dn_6ch_p01/best.pt")
    ap.add_argument("--baseline_ckpt", default=None, help="optional baseline ckpt path, e.g., checkpoints/best.pt")
    ap.add_argument("--outdir", default="outputs/plots", help="output folder for pngs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save_metrics", action="store_true", help="also save a small metrics json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Denoise model plots
    scores_dn, labels = compute_frame_scores(args.npz, args.ckpt, batch=args.batch)
    auc_dn = plot_roc(scores_dn, labels, outdir / "roc_frame_baseline.png", "ROC (frame_baseline) - Denoising AE")
    pr_dn = plot_pr(scores_dn, labels, outdir / "pr_frame_baseline.png", "PR (frame_baseline) - Denoising AE")
    plot_hist(scores_dn, labels, outdir / "hist_frame_baseline.png", "Score distribution (frame_baseline) - Denoising AE")

    metrics = {
        "denoise": {"ckpt": args.ckpt, "roc_auc": auc_dn, "pr_auc": pr_dn},
    }

    # Optional baseline comparison
    if args.baseline_ckpt:
        scores_b, labels_b = compute_frame_scores(args.npz, args.baseline_ckpt, batch=args.batch)
        auc_b = plot_roc(scores_b, labels_b, outdir / "roc_frame_baseline_baseline.png", "ROC (frame_baseline) - Baseline AE")
        pr_b = plot_pr(scores_b, labels_b, outdir / "pr_frame_baseline_baseline.png", "PR (frame_baseline) - Baseline AE")
        plot_hist(scores_b, labels_b, outdir / "hist_frame_baseline_baseline.png", "Score distribution (frame_baseline) - Baseline AE")

        plot_auc_compare(auc_b, auc_dn, outdir / "auc_compare.png", "Baseline vs Denoising (frame_baseline ROC-AUC)")

        metrics["baseline"] = {"ckpt": args.baseline_ckpt, "roc_auc": auc_b, "pr_auc": pr_b}

    if args.save_metrics:
        with open(outdir / "plot_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("[OK] Plots saved to:", str(outdir))
    for k, v in metrics.items():
        print(f"  - {k}: ROC-AUC={v['roc_auc']:.4f}, PR-AUC={v['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
