from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def residual_map(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    return (x - xhat).abs()


@torch.no_grad()
def score_mse(x: torch.Tensor, xhat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    err = (x - xhat) ** 2  # (B,8,64,64)
    frame_mse = err.mean(dim=(1, 2, 3))   # (B,)
    plane_mse = err.mean(dim=(2, 3))      # (B,8)
    return frame_mse, plane_mse


def signature_tags(plane_mse: torch.Tensor, ratio_thr: float = 3.0) -> Dict[str, object]:
    pm = plane_mse.detach().cpu()
    maxv = float(pm.max().item())
    meanv = float(pm.mean().item()) + 1e-12
    ratio = maxv / meanv
    return {
        "S1_one_plane_dominates": bool(ratio > ratio_thr),
        "ratio": float(ratio),
        "max_plane": int(pm.argmax().item()),
    }
