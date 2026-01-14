#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_residual.py

Visualize residual heatmaps for TRNG AE (ConvAE) on a given NPZ dataset.
Supports:
  - Plot a single sample (by --idx) for a given --plane
  - Auto-pick top-K highest frame_baseline scores and dump residual images

Examples:
  # Single sample, abs residual (|x-xhat|)
  python plot_residual.py --npz data/eval_mix.npz --ckpt checkpoints_dn_6ch_p01/best.pt --idx 0 --plane 3 --abs --out outputs/residual_abs_idx0_p3.png

  # Auto-pick top 5 highest-score samples, save in a folder (default: outputs/residual_topk)
  python plot_residual.py --npz data/eval_mix.npz --ckpt checkpoints_dn_6ch_p01/best.pt --topk 5 --plane 3 --abs --outdir outputs/residual_topk

  # Auto-pick top 5 among anomalies only
  python plot_residual.py --npz data/eval_mix.npz --ckpt checkpoints_dn_6ch_p01/best.pt --topk 5 --only_anomaly --plane 3 --abs --outdir outputs/residual_topk_anom

Notes:
  - "frame_baseline score" here = mean MSE across planes and pixels (same as eval.py's frame_baseline).
  - Residual heatmap can be abs(|x-xhat|) or squared((x-xhat)^2).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# -------------------------
# Model (must match train.py)
# -------------------------
class ConvAE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def _pick_npz(d, names):
    for n in names:
        if n in d.files:
            return d[n]
    return None


def _safe_meta(meta_item):
    """meta_item may be bytes/str/json; return dict if possible else {}"""
    if meta_item is None:
        return {}
    try:
        m = meta_item
        if isinstance(m, (bytes, bytearray)):
            m = m.decode("utf-8", errors="ignore")
        if isinstance(m, str):
            return json.loads(m)
        if isinstance(m, dict):
            return m
    except Exception:
        pass
    return {}


@torch.no_grad()
def load_model(ckpt_path: str, in_ch: int):
    ck = torch.load(ckpt_path, map_location="cpu")
    ck_in = int(ck.get("in_ch", in_ch))
    if ck_in != in_ch:
        raise RuntimeError(f"Channel mismatch: ckpt expects {ck_in} channels, but data has {in_ch}. "
                           f"Use a matching train/eval channel setup.")
    model = ConvAE(in_ch=ck_in)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


@torch.no_grad()
def compute_frame_scores(model, device, X: np.ndarray, batch: int = 64):
    """
    frame_baseline score = mean MSE over planes/pixels per sample
    X: (N,P,H,W)
    returns: scores (N,)
    """
    N = X.shape[0]
    scores = np.zeros((N,), dtype=np.float32)
    i = 0
    while i < N:
        j = min(N, i + batch)
        xb = torch.from_numpy(X[i:j]).to(device)
        xhat = model(xb)
        res = xb - xhat
        res = (res * res).detach().cpu().numpy()  # (B,P,H,W)
        plane_mse = res.mean(axis=(2, 3))          # (B,P)
        s_frame = plane_mse.mean(axis=1)           # (B,)
        scores[i:j] = s_frame.astype(np.float32)
        i = j
    return scores


@torch.no_grad()
def compute_residual_map(model, device, x1: np.ndarray, plane: int, use_abs: bool):
    """
    x1: (P,H,W) single sample
    returns: residual (H,W) for given plane
    """
    xb = torch.from_numpy(x1[None, ...]).to(device)  # (1,P,H,W)
    xhat = model(xb)
    diff = (xb - xhat)[0, plane].detach().cpu().numpy()  # (H,W)
    if use_abs:
        return np.abs(diff), "|x-xhat|"
    return diff * diff, "(x-xhat)^2"


def save_residual_figure(residual_hw: np.ndarray, title: str, out_png: Path, vmin=None, vmax=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 6.5))
    im = plt.imshow(residual_hw, aspect="equal", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="e.g., data/eval_mix.npz")
    ap.add_argument("--ckpt", required=True, help="e.g., checkpoints_dn_6ch_p01/best.pt")
    ap.add_argument("--plane", type=int, default=0, help="channel/plane index")
    ap.add_argument("--idx", type=int, default=None, help="plot a single sample index (optional)")
    ap.add_argument("--topk", type=int, default=0, help="if >0, auto-pick top-K highest frame_baseline scores")
    ap.add_argument("--only_anomaly", action="store_true", help="when using --topk, restrict to anomaly (y!=0)")
    ap.add_argument("--only_healthy", action="store_true", help="when using --topk, restrict to healthy (y==0)")
    ap.add_argument("--abs", action="store_true", help="use |x-xhat| heatmap (default is squared error)")
    ap.add_argument("--batch", type=int, default=64, help="batch size for score computation")
    ap.add_argument("--out", default="outputs/residual.png", help="output png for single-sample mode")
    ap.add_argument("--outdir", default="outputs/residual_topk", help="output dir for top-k mode")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--save_index", action="store_true", help="in top-k mode, save a JSON index file")
    args = ap.parse_args()

    if args.only_anomaly and args.only_healthy:
        raise SystemExit("Choose only one of --only_anomaly or --only_healthy.")

    d = np.load(args.npz, allow_pickle=True)
    X = _pick_npz(d, ["x", "X", "data", "arr_0"])
    y = _pick_npz(d, ["y", "Y", "label", "labels", "arr_1"])
    meta = _pick_npz(d, ["meta", "Meta", "metadata", "arr_2"])

    if X is None:
        raise KeyError(f"Cannot find x in npz. keys={d.files}")
    X = X.astype(np.float32)

    if y is None:
        # allow plotting even if no labels; topk filtering won't work
        y = np.zeros((X.shape[0],), dtype=np.int64)
    else:
        y = y.astype(np.int64)

    N, P, H, W = X.shape
    if not (0 <= args.plane < P):
        raise SystemExit(f"--plane out of range. data has {P} planes (valid: 0..{P-1})")

    model, device = load_model(args.ckpt, in_ch=P)

    # -------------------------
    # Single-sample mode
    # -------------------------
    if args.idx is not None:
        idx = int(args.idx)
        if not (0 <= idx < N):
            raise SystemExit(f"--idx out of range. data has {N} samples (valid: 0..{N-1})")

        label_str = "healthy" if int(y[idx]) == 0 else "anomaly"
        inj = ""
        if meta is not None:
            mm = _safe_meta(meta[idx])
            inj = str(mm.get("inj_type", "")) if isinstance(mm, dict) else ""

        res_map, metric_name = compute_residual_map(model, device, X[idx], args.plane, use_abs=args.abs)
        title = f"Residual {metric_name} (plane={args.plane})  label={label_str}  inj={inj}  idx={idx}"
        out_png = Path(args.out)
        save_residual_figure(res_map, title, out_png, vmin=args.vmin, vmax=args.vmax)
        print("[OK] Saved:", str(out_png))
        return

    # -------------------------
    # Top-k mode
    # -------------------------
    if args.topk and args.topk > 0:
        scores = compute_frame_scores(model, device, X, batch=args.batch)

        # filter indices if requested
        idxs = np.arange(N)
        if args.only_anomaly:
            idxs = idxs[y != 0]
        if args.only_healthy:
            idxs = idxs[y == 0]

        if idxs.size == 0:
            raise SystemExit("No samples after filtering (check labels / filters).")

        # select top-k by score
        sel = idxs[np.argsort(scores[idxs])[::-1]]
        sel = sel[: int(args.topk)]

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        index_rows = []
        for rank, idx in enumerate(sel, start=1):
            label_str = "healthy" if int(y[idx]) == 0 else "anomaly"
            inj = ""
            if meta is not None:
                mm = _safe_meta(meta[idx])
                inj = str(mm.get("inj_type", "")) if isinstance(mm, dict) else ""

            res_map, metric_name = compute_residual_map(model, device, X[idx], args.plane, use_abs=args.abs)
            title = (f"Residual {metric_name} (plane={args.plane})  label={label_str}  inj={inj}  "
                     f"idx={idx}  rank={rank}/{len(sel)}  frame_score={float(scores[idx]):.6f}")

            out_png = outdir / f"residual_rank{rank:02d}_idx{idx}_p{args.plane}.png"
            save_residual_figure(res_map, title, out_png, vmin=args.vmin, vmax=args.vmax)

            index_rows.append({
                "rank": rank,
                "idx": int(idx),
                "plane": int(args.plane),
                "label": label_str,
                "inj_type": inj,
                "frame_score": float(scores[idx]),
                "png": str(out_png).replace("\\", "/"),
                "metric": metric_name,
            })

        if args.save_index:
            with open(outdir / "index.json", "w", encoding="utf-8") as f:
                json.dump(index_rows, f, indent=2)

        print(f"[OK] Saved {len(sel)} residual plots to: {str(outdir)}")
        if args.save_index:
            print(f"[OK] Saved index: {str(outdir / 'index.json')}")
        return

    raise SystemExit("Nothing to do. Provide either --idx for single sample, or --topk K for top-k export.")


if __name__ == "__main__":
    main()
