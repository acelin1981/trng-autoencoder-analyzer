#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Dataset
# =========================
class NPZDataset(Dataset):
    def __init__(self, npz_path, use_only_healthy=False):
        d = np.load(npz_path, allow_pickle=True)

        def pick(names):
            for n in names:
                if n in d.files:
                    return d[n]
            return None

        x = pick(["x", "X", "data", "Data", "arr_0"])
        y = pick(["y", "Y", "label", "labels", "arr_1"])

        if x is None:
            raise KeyError(
                f"Cannot find data array in npz. Available keys={d.files}"
            )

        self.x = x.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)

        # healthy-only mode
        if use_only_healthy and self.y is not None:
            mask = (self.y == 0)
            self.x = self.x[mask]
            self.y = self.y[mask]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

# =========================
# Model
# =========================
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

# =========================
# Training
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--noise_p", type=float, default=0.01)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = NPZDataset(args.npz, use_only_healthy=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    in_ch = ds.x.shape[1]
    model = ConvAE(in_ch=in_ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb in loader:
            xb = xb.to(device)

            if args.denoise:
                noise = (torch.rand_like(xb) < args.noise_p).float()
                xb_in = xb * (1.0 - noise)
            else:
                xb_in = xb

            xhat = model(xb_in)
            loss = ((xb - xhat) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        m = float(np.mean(losses))
        print(f"[Epoch {ep:03d}] loss={m:.6f}")

        if m < best_loss:
            best_loss = m
            torch.save(
                {"model": model.state_dict(), "in_ch": in_ch},
                outdir / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "in_ch": in_ch},
        outdir / "last.pt",
    )

    print(f"[OK] Training finished. Best loss={best_loss:.6f}")

if __name__ == "__main__":
    main()
