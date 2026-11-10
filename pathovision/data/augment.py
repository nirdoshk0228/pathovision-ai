"""Histology-aware augmentations (numpy, applied jointly to image and mask).

* Dihedral group D4 (4 rotations x optional flip): tissue has no canonical
  orientation, so all 8 variants are equally valid. The same 8 transforms are
  reused for test-time augmentation (TTA).
* HED stain jitter: simulates stain variation between labs.
* Mild brightness/contrast and blur: simulates scanner focus/exposure.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from pathovision.preprocessing.stain import hed_jitter


def dihedral(img: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rotate by 90*(k%4) degrees, flip horizontally if k >= 4."""
    out = np.rot90(img, k % 4, axes=(0, 1))
    if k >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def dihedral_inverse(img: np.ndarray, k: int) -> np.ndarray:
    out = img[:, ::-1] if k >= 4 else img
    return np.ascontiguousarray(np.rot90(out, -(k % 4), axes=(0, 1)))


class Augmenter:
    def __init__(self, stain: bool = True, color: bool = True, blur_p: float = 0.1, seed: Optional[int] = None):
        self.stain, self.color, self.blur_p = stain, color, blur_p
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        k = int(self.rng.integers(0, 8))
        img = dihedral(img, k)
        if mask is not None:
            mask = dihedral(mask, k)
        if self.stain and self.rng.random() < 0.8:
            img = hed_jitter(img, sigma=0.05, bias=0.03, rng=self.rng)
        if self.color and self.rng.random() < 0.5:
            a = self.rng.uniform(0.9, 1.1)
            b = self.rng.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if self.rng.random() < self.blur_p:
            img = cv2.GaussianBlur(img, (3, 3), self.rng.uniform(0.3, 1.0))
        return img, mask
