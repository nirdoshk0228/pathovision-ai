"""Rendering helpers: heatmap colouring, overlays, contours."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def colorize(prob: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Probability in [0, 1] -> RGB uint8."""
    u8 = np.clip(prob * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB)


def resize_to(img: np.ndarray, hw: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (hw[1], hw[0]), interpolation=interp)


def overlay_heatmap(thumb: np.ndarray, prob: np.ndarray, valid: Optional[np.ndarray] = None,
                    alpha: float = 0.45, min_prob: float = 0.0) -> np.ndarray:
    """Blend a probability map over the thumbnail, only where ``valid`` (e.g. tissue) and prob >= min_prob."""
    h, w = thumb.shape[:2]
    p = resize_to(prob.astype(np.float32), (h, w))
    color = colorize(p)
    m = np.ones((h, w), bool) if valid is None else resize_to(valid.astype(np.uint8), (h, w), nearest=True) > 0
    m &= p >= min_prob
    out = thumb.copy()
    out[m] = (alpha * color[m] + (1 - alpha) * thumb[m]).astype(np.uint8)
    return out


def draw_mask_contours(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    m = resize_to(mask.astype(np.uint8), (h, w), nearest=True)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    m = resize_to(mask.astype(np.uint8), (h, w), nearest=True) > 0
    out = img.copy()
    out[m] = (alpha * np.array(color) + (1 - alpha) * img[m]).astype(np.uint8)
    return out


def add_colorbar(img: np.ndarray, height: int = 18, label: str = "tumour probability") -> np.ndarray:
    w = img.shape[1]
    bar = colorize(np.tile(np.linspace(0, 1, w)[None], (height, 1)))
    pad = np.full((height + 16, w, 3), 255, np.uint8)
    pad[:height] = bar
    cv2.putText(pad, "0", (2, height + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(pad, "1", (w - 10, height + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(pad, label, (w // 2 - 60, height + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return np.vstack([img, pad])
