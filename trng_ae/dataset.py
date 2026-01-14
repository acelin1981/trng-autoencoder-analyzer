# trng_ae/dataset.py
from __future__ import annotations

from typing import Any, Dict, Tuple, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_primitive(v: Any) -> Any:
    """Convert values into DataLoader-collatable primitives.
    - None -> ""
    - numpy scalars -> python scalars
    - dict/list/ndarray -> JSON string (prevents nested-dict collate KeyError)
    - others -> keep if already primitive
    """
    if v is None:
        return ""
    # numpy scalar
    if isinstance(v, (np.generic,)):
        return v.item()
    # dict/list/tuple/ndarray -> stringify to stop recursive collate
    if isinstance(v, (dict, list, tuple, np.ndarray)):
        try:
            return json.dumps(v, sort_keys=True, default=str)
        except Exception:
            return str(v)
    # torch tensor shouldn't appear in meta; if it does, stringify
    if torch.is_tensor(v):
        return str(v.detach().cpu().numpy())
    return v


class BitPlaneDataset(Dataset):
    """
    Loads bit-plane windows and per-sample metadata from an .npz.

    NPZ keys:
      - X: (N, 8, 64, 64) float/uint8 {0,1}
      - meta: (N,) array of dict-like objects (pickle)
    """

    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        if "X" not in z or "meta" not in z:
            raise KeyError("NPZ must contain keys: 'X' and 'meta'")

        self.X = z["X"].astype(np.float32)
        self.meta: List[Dict[str, Any]] = [dict(m) for m in z["meta"]]

        # Fixed schema (only what we need downstream). Missing keys will be filled.
        self._defaults: Dict[str, Any] = {
            "label": 0,
            "inj_type": "",
            "plane": -1,
            "severity": 0.0,
        }

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = torch.from_numpy(self.X[idx])  # (8,64,64)
        raw = dict(self.meta[idx])

        # Build a SAFE meta dict with fixed keys only (avoid nested dict surprises)
        m: Dict[str, Any] = {}
        for k, dv in self._defaults.items():
            m[k] = _to_primitive(raw.get(k, dv))

        # If you still want to keep extra fields for debugging, stringify them:
        # (optional) uncomment this block
        # for k, v in raw.items():
        #     if k not in m:
        #         m[k] = _to_primitive(v)

        return x, m
