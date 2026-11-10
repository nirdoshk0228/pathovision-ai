"""Shared training/inference loops."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def tta_logits(model: torch.nn.Module, x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Average logits over up to 8 dihedral transforms (classification: output is invariant)."""
    outs = []
    for k in range(n):
        xi = torch.rot90(x, k % 4, dims=(2, 3))
        if k >= 4:
            xi = torch.flip(xi, dims=(3,))
        outs.append(model(xi))
    return torch.stack(outs).mean(0)


def tta_seg_logits(model: torch.nn.Module, x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Segmentation TTA: transform input, predict, invert transform on the output map, average."""
    outs = []
    for k in range(n):
        xi = torch.rot90(x, k % 4, dims=(2, 3))
        if k >= 4:
            xi = torch.flip(xi, dims=(3,))
        y = model(xi)
        if k >= 4:
            y = torch.flip(y, dims=(3,))
        outs.append(torch.rot90(y, -(k % 4), dims=(2, 3)))
    return torch.stack(outs).mean(0)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device, tta: int = 1,
                  amp: bool = True):
    """Returns (probs, labels) for a classification loader."""
    model.eval()
    probs, labels = [], []
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = tta_logits(model, x, tta) if tta > 1 else model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker an independent augmentation RNG.

    Without this, forked workers inherit identical RNG state and apply the *same*
    'random' augmentation to different samples.
    """
    info = torch.utils.data.get_worker_info()
    seed = int(torch.initial_seed() % (2 ** 32))
    np.random.seed(seed)
    aug = getattr(info.dataset, "augment", None) if info is not None else None
    if aug is not None and hasattr(aug, "rng"):
        aug.rng = np.random.default_rng(seed)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best: Optional[float] = None
        self.bad = 0

    def step(self, value: float) -> bool:
        """Returns True if this is a new best."""
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return better

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience
