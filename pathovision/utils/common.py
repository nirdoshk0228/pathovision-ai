"""Small shared helpers: seeding, device selection, checkpoints, tensor conversion."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(batch_hwc_uint8: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) or (H, W, 3) uint8 -> ImageNet-normalised float tensor (B, 3, H, W)."""
    x = batch_hwc_uint8
    if x.ndim == 3:
        x = x[None]
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def save_checkpoint(path: str, model: torch.nn.Module, meta: Dict[str, Any]) -> None:
    """Weights + the preprocessing contract (arch, patch size, mpp, threshold...).

    Storing the contract inside the checkpoint means inference can never silently
    run a 10x model on 20x patches.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location="cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt["state_dict"], ckpt.get("meta", {})


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self):
        self.stages: Dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.stages.items()}


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Linux/macOS)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if os.uname().sysname == "Linux" else r / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover
        import psutil

        return psutil.Process().memory_info().rss / 1e6
