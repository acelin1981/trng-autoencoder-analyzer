from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np


@dataclass
class FrameMeta:
    """Metadata for a generated sample/frame."""
    label: str  # "healthy" or "degraded"
    inj_type: Optional[str]  # e.g., "bias", "stuck_plane", ...
    params: Dict


def _rng(seed: int):
    return np.random.default_rng(seed)


def generate_healthy_bitplanes(seed: int, p: float = 0.5, plane_bias_jitter: float = 0.01) -> np.ndarray:
    """
    Generate a 'healthy' 8x64x64 bit-plane tensor.
    We keep it close to Bernoulli(0.5) but allow tiny per-plane bias jitter
    (to create a learnable baseline without claiming physical causality).
    """
    r = _rng(seed)
    x = np.zeros((8, 64, 64), dtype=np.float32)
    for b in range(8):
        pb = float(np.clip(p + r.uniform(-plane_bias_jitter, plane_bias_jitter), 0.45, 0.55))
        x[b] = (r.random((64, 64)) < pb).astype(np.float32)
    return x


# ------------------------- Error injection patterns -------------------------

def inject_bias(x: np.ndarray, seed: int, planes: List[int] = [0, 1, 2, 3], delta: float = 0.08) -> Tuple[np.ndarray, FrameMeta]:
    """
    Increase probability of 1s for selected planes by flipping some 0s->1s (bias up).
    delta roughly controls flip rate.
    """
    r = _rng(seed)
    y = x.copy()
    for b in planes:
        zeros = (y[b] < 0.5)
        flip = (r.random((64, 64)) < delta) & zeros
        y[b][flip] = 1.0
    return y, FrameMeta(label="degraded", inj_type="bias", params={"planes": planes, "delta": delta})


def inject_stuck_plane(x: np.ndarray, seed: int, plane: int = 0, value: float = 1.0) -> Tuple[np.ndarray, FrameMeta]:
    """Force one entire bit-plane to a constant (stuck-at pattern)."""
    y = x.copy()
    y[plane][:] = float(value)
    return y, FrameMeta(label="degraded", inj_type="stuck_plane", params={"plane": plane, "value": value})


def inject_block_stuck(x: np.ndarray, seed: int, plane: int = 0, block: Tuple[int, int, int, int] = (16, 16, 32, 32), value: float = 0.0) -> Tuple[np.ndarray, FrameMeta]:
    """Force a rectangular block in a plane to a constant."""
    y = x.copy()
    y0, x0, h, w = block
    y[plane][y0:y0+h, x0:x0+w] = float(value)
    return y, FrameMeta(label="degraded", inj_type="block_stuck", params={"plane": plane, "block": block, "value": value})


def inject_stripes(x: np.ndarray, seed: int, plane: int = 1, period: int = 4, strength: float = 0.6) -> Tuple[np.ndarray, FrameMeta]:
    """
    Add periodic vertical stripes by forcing some columns to 1 with probability 'strength'.
    This mimics a data-path/periodic artifact (phenomenological).
    """
    r = _rng(seed)
    y = x.copy()
    mask = np.zeros((64, 64), dtype=bool)
    for c in range(0, 64, period):
        mask[:, c] = True
    push = (r.random((64, 64)) < strength) & mask
    y[plane][push] = 1.0
    return y, FrameMeta(label="degraded", inj_type="stripes", params={"plane": plane, "period": period, "strength": strength})


def inject_burst_errors(x: np.ndarray, seed: int, n_bursts: int = 3, burst_size: int = 64) -> Tuple[np.ndarray, FrameMeta]:
    """
    Randomly flip bits in short bursts across random planes.
    """
    r = _rng(seed)
    y = x.copy()
    for _ in range(n_bursts):
        plane = int(r.integers(0, 8))
        ys = int(r.integers(0, 64))
        xs = int(r.integers(0, 64))
        for _k in range(burst_size):
            yy = (ys + int(r.integers(0, 7))) % 64
            xx = (xs + int(r.integers(0, 7))) % 64
            y[plane, yy, xx] = 1.0 - y[plane, yy, xx]
    return y, FrameMeta(label="degraded", inj_type="burst", params={"n_bursts": n_bursts, "burst_size": burst_size})


INJECTORS = {
    "bias": inject_bias,
    "stuck_plane": inject_stuck_plane,
    "block_stuck": inject_block_stuck,
    "stripes": inject_stripes,
    "burst": inject_burst_errors,
}


def make_sample(seed: int, degraded: bool, inj: Optional[str]) -> Tuple[np.ndarray, FrameMeta]:
    x = generate_healthy_bitplanes(seed)
    if not degraded:
        return x, FrameMeta(label="healthy", inj_type=None, params={})
    if inj is None:
        inj = random.choice(list(INJECTORS.keys()))
    y, meta = INJECTORS[inj](x, seed=seed + 999)
    return y, meta


def generate_dataset_npz(
    out_path: str,
    n_healthy: int,
    n_degraded: int,
    seed: int = 1234,
    degraded_mix: Optional[List[str]] = None,
) -> None:
    """
    Generate an .npz file with:
      X: float32 array (N, 8, 64, 64)
      meta: list of dicts length N
    """
    r = _rng(seed)
    X_list = []
    meta_list = []

    for _i in range(n_healthy):
        s = int(r.integers(0, 2**31 - 1))
        x, meta = make_sample(s, degraded=False, inj=None)
        X_list.append(x)
        meta_list.append(meta.__dict__)

    if degraded_mix is None:
        degraded_mix = list(INJECTORS.keys())
    for i in range(n_degraded):
        s = int(r.integers(0, 2**31 - 1))
        inj = degraded_mix[i % len(degraded_mix)]
        x, meta = make_sample(s, degraded=True, inj=inj)
        X_list.append(x)
        meta_list.append(meta.__dict__)

    X = np.stack(X_list, axis=0).astype(np.float32)
    np.savez_compressed(out_path, X=X, meta=np.array(meta_list, dtype=object))
