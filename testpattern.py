
"""
testpattern.py

Generate synthetic TRNG byte frames (healthy + anomalies), train the CNN autoencoder on healthy data,
and produce a JPEG report with:
  1) Training curve (MSE)
  2) Anomaly score distributions by pattern
  3) Example residual heatmaps (per-bitplane)

Outputs:
  - ./outputs/test_patterns.npz   (uint8 frames)
  - ./outputs/model.pt           (trained model)
  - ./outputs/report.jpeg        (plots)

Run:
  python testpattern.py

Optional:
  python testpattern.py --frames 400 --epochs 12 --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from autoencoder import (
    ConvAutoencoder,
    TRNGBitplaneDataset,
    TrainConfig,
    train_autoencoder,
    anomaly_scores_mse,
    reconstruct_one,
)


# ----------------------------
# Synthetic patterns (uint8 frames, shape (N,64,64))
# ----------------------------

def rng_bytes(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    return rng.integers(0, 256, size=(n_frames, 64, 64), dtype=np.uint8)


def stuck_bit(rng: np.random.Generator, n_frames: int, bit: int = 0, value: int = 0) -> np.ndarray:
    x = rng_bytes(rng, n_frames)
    mask = np.uint8(1 << bit)
    if value == 0:
        x = (x & (~mask)).astype(np.uint8)
    else:
        x = (x | mask).astype(np.uint8)
    return x


def biased_bit(rng: np.random.Generator, n_frames: int, bit: int = 0, p_one: float = 0.8) -> np.ndarray:
    """
    Force a single bit to have P(1)=p_one by sampling that bit and merging into random bytes.
    """
    x = rng_bytes(rng, n_frames)
    mask = np.uint8(1 << bit)
    ones = (rng.random(size=(n_frames, 64, 64)) < p_one).astype(np.uint8)
    x = (x & (~mask)).astype(np.uint8)  # clear bit
    x = (x | (ones * mask)).astype(np.uint8)
    return x


def repeating_block(rng: np.random.Generator, n_frames: int, block: int = 32) -> np.ndarray:
    """
    Make repetition: each row repeats every 'block' pixels (introduces structure).
    """
    x = rng_bytes(rng, n_frames)
    for i in range(n_frames):
        base = x[i, :, :block].copy()
        reps = 64 // block
        x[i] = np.concatenate([base] * reps, axis=1)
    return x


def low_entropy_lsb(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    """
    Make LSBs low-entropy by taking LSB from a slowly changing counter.
    """
    x = rng_bytes(rng, n_frames)
    counter = np.arange(n_frames * 64 * 64, dtype=np.uint32).reshape(n_frames, 64, 64)
    lsb = (counter & 1).astype(np.uint8)
    x = (x & 0xFE).astype(np.uint8)  # clear LSB
    x = (x | lsb).astype(np.uint8)
    return x


def build_dataset(rng: np.random.Generator, n_frames: int) -> Dict[str, np.ndarray]:
    return {
        "healthy": rng_bytes(rng, n_frames),
        "stuck_bit0_0": stuck_bit(rng, n_frames, bit=0, value=0),
        "stuck_bit7_1": stuck_bit(rng, n_frames, bit=7, value=1),
        "biased_bit0_p0p8": biased_bit(rng, n_frames, bit=0, p_one=0.8),
        "repeating_block32": repeating_block(rng, n_frames, block=32),
        "low_entropy_lsb": low_entropy_lsb(rng, n_frames),
    }


# ----------------------------
# Plot report
# ----------------------------

def plot_report(
    out_jpeg: Path,
    train_losses,
    val_losses,
    scores_by_name: Dict[str, np.ndarray],
    residual_examples: Dict[str, np.ndarray],
):
    """
    residual_examples[name] = residual bitplanes [8,64,64] for one sample
    """
    out_jpeg.parent.mkdir(parents=True, exist_ok=True)

    # Layout: 2 rows x 3 cols.
    fig = plt.figure(figsize=(16, 9))

    # 1) Train curve
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.plot(train_losses, label="train")
    if val_losses:
        ax1.plot(val_losses, label="val")
    ax1.set_title("Autoencoder training loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Score distributions
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    names = list(scores_by_name.keys())
    data = [scores_by_name[k] for k in names]
    ax2.boxplot(data, labels=names, showfliers=False)
    ax2.set_title("Anomaly score by pattern (per-sample MSE)")
    ax2.set_ylabel("MSE")
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(True, axis="y", alpha=0.3)

    # 3-6) Residual heatmaps for a few patterns (show mean over bitplanes for compactness)
    show_keys = ["healthy", "stuck_bit0_0", "biased_bit0_p0p8", "repeating_block32"]
    for idx, key in enumerate(show_keys):
        r = residual_examples[key]  # [8,64,64]
        mean_r = r.mean(axis=0)     # [64,64]
        ax = plt.subplot2grid((2, 3), (1, idx % 3))
        im = ax.imshow(mean_r, interpolation="nearest")
        ax.set_title(f"Residual (mean over 8 planes): {key}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("TRNG Anomaly Detection with CNN Autoencoder (bit-plane residuals)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_jpeg, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300, help="Frames per pattern (each frame is 64x64 bytes).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build patterns
    ds_dict = build_dataset(rng, args.frames)
    np.savez_compressed(out_dir / "test_patterns.npz", **ds_dict)

    # Train split for healthy
    healthy = ds_dict["healthy"]
    n = healthy.shape[0]
    n_train = int(n * 0.8)
    train_frames = healthy[:n_train]
    val_frames = healthy[n_train:]

    train_ds = TRNGBitplaneDataset(train_frames)
    val_ds = TRNGBitplaneDataset(val_frames)

    # Train
    model = ConvAutoencoder()
    cfg = TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr)
    model, train_losses, val_losses = train_autoencoder(model, train_ds, cfg, val_ds=val_ds)

    # Save model
    model_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, model_path)

    # Scores on each pattern
    scores_by = {}
    residual_examples = {}

    for name, frames in ds_dict.items():
        dset = TRNGBitplaneDataset(frames)
        scores = anomaly_scores_mse(model, dset, batch_size=args.batch, device=cfg.device)
        scores_by[name] = scores

        # Pick one example and store residuals for plot
        x0 = dset[0]  # tensor [8,64,64]
        _, _, resid = reconstruct_one(model, x0, device=cfg.device)
        residual_examples[name] = resid

    # Plot report jpeg
    out_jpeg = out_dir / "report.jpeg"
    plot_report(out_jpeg, train_losses, val_losses, scores_by, residual_examples)

    # Print a compact summary
    print("Saved:")
    print(f" - {out_dir/'test_patterns.npz'}")
    print(f" - {model_path}")
    print(f" - {out_jpeg}")
    print("\nMedian anomaly scores (lower is better):")
    for k, v in scores_by.items():
        print(f"  {k:18s}  median={np.median(v):.6f}  mean={np.mean(v):.6f}")


if __name__ == "__main__":
    main()
