
"""
autoencoder.py

CNN Autoencoder for TRNG anomaly detection (bit-plane images) based on the architecture
described in the uploaded slide deck (Input [8,64,64], Conv/MaxPool encoder, ConvTranspose decoder).  fileciteturn0file0L1-L14

- Input: 8 bit-planes (LSB..MSB) from 64x64 bytes (4096 bytes) -> tensor [8,64,64]
- Loss: MSE reconstruction loss
- Anomaly score: per-sample MSE (higher = more suspicious)

This module is intentionally self-contained and can be imported by testpattern.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utilities: bytes -> bit-planes
# ----------------------------

def bytes_to_bitplanes_u8(frame_u8: np.ndarray) -> np.ndarray:
    """
    Convert a (64,64) uint8 array into 8 bit-plane images shaped (8,64,64).
    Plane 0 is LSB, plane 7 is MSB. Values are float32 in [0,1].
    """
    if frame_u8.dtype != np.uint8:
        frame_u8 = frame_u8.astype(np.uint8, copy=False)
    if frame_u8.shape != (64, 64):
        raise ValueError(f"Expected frame shape (64,64), got {frame_u8.shape}")

    planes = np.empty((8, 64, 64), dtype=np.float32)
    for b in range(8):
        planes[b] = ((frame_u8 >> b) & 1).astype(np.float32)
    return planes


def make_frames_from_bytes(byte_stream: np.ndarray) -> np.ndarray:
    """
    Turn a 1D uint8 stream into N frames of shape (64,64) by chunking 4096 bytes per frame.
    """
    if byte_stream.dtype != np.uint8:
        byte_stream = byte_stream.astype(np.uint8, copy=False)
    chunk = 64 * 64
    n = byte_stream.size // chunk
    if n <= 0:
        raise ValueError("Need at least 4096 bytes to form one frame (64*64).")
    trimmed = byte_stream[: n * chunk]
    frames = trimmed.reshape(n, 64, 64)
    return frames


# ----------------------------
# Dataset
# ----------------------------

class TRNGBitplaneDataset(Dataset):
    """
    Dataset of bit-plane tensors. Backed by:
      - npy/npz file containing frames of bytes (N,64,64) uint8
      - or already-prepared float bitplanes (N,8,64,64)
    """

    def __init__(self, data: np.ndarray):
        """
        data:
          - uint8 frames: (N,64,64)
          - float bitplanes: (N,8,64,64) in [0,1]
        """
        if data.ndim == 3 and data.shape[1:] == (64, 64):
            self.frames_u8 = data.astype(np.uint8, copy=False)
            self.bitplanes = None
        elif data.ndim == 4 and data.shape[1:] == (8, 64, 64):
            self.frames_u8 = None
            self.bitplanes = data.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

    def __len__(self) -> int:
        return int(self.frames_u8.shape[0] if self.frames_u8 is not None else self.bitplanes.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.bitplanes is not None:
            x = self.bitplanes[idx]
        else:
            x = bytes_to_bitplanes_u8(self.frames_u8[idx])
        return torch.from_numpy(x)  # [8,64,64]


def load_frames(path: Path, key: Optional[str] = None) -> np.ndarray:
    """
    Load uint8 frames (N,64,64) from .npy or .npz.
    If .npz and key not provided, use the first array.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        if key is None:
            key = list(z.keys())[0]
        return z[key]
    raise ValueError("Supported formats: .npy or .npz")


# ----------------------------
# Model: CNN Autoencoder (as per slides)
# ----------------------------

class ConvAutoencoder(nn.Module):
    """
    Encoder:
      Conv(8→32,3x3)+ReLU
      Conv(32→32,3x3)+ReLU
      MaxPool(2) -> [32,32,32]
      Conv(32→64,3x3)+ReLU
      Conv(64→64,3x3)+ReLU
      MaxPool(2) -> [64,16,16]
      Conv(64→128,3x3)+ReLU -> bottleneck [128,16,16]
    Decoder:
      ConvTranspose(128→64,2x2,stride=2) -> [64,32,32]
      Conv(64→64,3x3)+ReLU
      ConvTranspose(64→32,2x2,stride=2) -> [32,64,64]
      Conv(32→32,3x3)+ReLU
      Conv(32→8,3x3)+Sigmoid -> [8,64,64]
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32->64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        y = self.dec(z)
        return y


# ----------------------------
# Training / Evaluation
# ----------------------------

@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


def train_autoencoder(
    model: nn.Module,
    train_ds: Dataset,
    cfg: TrainConfig,
    val_ds: Optional[Dataset] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction="mean")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    train_losses, val_losses = [], []
    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for x in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = model(x)
            loss = loss_fn(y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_losses.append(running / max(n, 1))

        if val_loader is not None:
            model.eval()
            running = 0.0
            n = 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(cfg.device, non_blocking=True)
                    y = model(x)
                    loss = loss_fn(y, x)
                    running += float(loss.item()) * x.size(0)
                    n += x.size(0)
            val_losses.append(running / max(n, 1))

    return model, train_losses, val_losses


@torch.no_grad()
def anomaly_scores_mse(model: nn.Module, ds: Dataset, batch_size: int = 64, device: Optional[str] = None) -> np.ndarray:
    """
    Per-sample MSE reconstruction error.
    Returns shape (N,) float64.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores: List[float] = []
    for x in loader:
        x = x.to(device)
        y = model(x)
        mse = (y - x).pow(2).mean(dim=(1, 2, 3))  # per-sample
        scores.extend(mse.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def reconstruct_one(model: nn.Module, x: torch.Tensor, device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct one sample x [8,64,64].
    Returns (x_np, y_np, residual_np) as float arrays [8,64,64].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    xin = x.unsqueeze(0).to(device)
    y = model(xin)[0].detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    residual = np.abs(y - x_np)
    return x_np, y, residual


def main():
    p = argparse.ArgumentParser(description="Train/evaluate TRNG CNN autoencoder (bit-plane).")
    p.add_argument("--train", type=str, required=True, help="Path to .npy/.npz of uint8 frames (N,64,64) for training.")
    p.add_argument("--val", type=str, default="", help="Optional val set path.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="model.pt", help="Output model path.")
    args = p.parse_args()

    train_frames = load_frames(Path(args.train))
    train_ds = TRNGBitplaneDataset(train_frames)

    val_ds = None
    if args.val:
        val_frames = load_frames(Path(args.val))
        val_ds = TRNGBitplaneDataset(val_frames)

    cfg = TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr)
    model = ConvAutoencoder()
    model, tr, va = train_autoencoder(model, train_ds, cfg, val_ds=val_ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, out_path)

    print(f"Saved: {out_path}")
    print(f"Train losses: {tr[-5:]}")
    if va:
        print(f"Val losses: {va[-5:]}")


if __name__ == "__main__":
    main()
