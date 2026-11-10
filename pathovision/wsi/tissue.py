"""Tissue detection on a low-resolution slide overview.

Typically 70-90% of a whole-slide image is glass. Detecting tissue on a ~2k px
thumbnail first means the expensive deep-learning stage only touches patches
that contain cells, which is the single biggest speed-up in the pipeline.

Method: H&E-stained tissue is *coloured* (pink/purple) while glass is white/grey,
so saturation in HSV space separates them well. Otsu picks the threshold per
slide, which makes it robust to scanner and staining differences. Morphology then
removes dust and fills small holes.
"""
from __future__ import annotations

import cv2
import numpy as np


def tissue_mask(
    rgb: np.ndarray,
    min_saturation: int = 15,
    max_value_background: int = 235,
    min_object_frac: float = 0.0005,
    kernel_frac: float = 0.004,
    remove_dark: bool = True,
) -> np.ndarray:
    """Boolean tissue mask for an RGB overview image.

    Args:
        rgb: HxWx3 uint8 thumbnail.
        min_saturation: floor for the Otsu threshold (prevents a blank slide from
            producing a near-zero threshold that marks everything as tissue).
        max_value_background: pixels brighter than this in every channel are glass.
        min_object_frac: connected components smaller than this fraction of the
            image are discarded (dust, debris).
        kernel_frac: morphology kernel size relative to the image's short side.
        remove_dark: drop near-black pixels (pen marks, slide edges, folds).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu), float(min_saturation))
    mask = sat > thr

    mask &= ~(rgb.min(axis=2) > max_value_background)
    if remove_dark:
        mask &= ~(rgb.max(axis=2) < 40)

    k = max(3, int(round(min(rgb.shape[:2]) * kernel_frac)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = min_object_frac * m.size
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[labels]

    return fill_holes(m)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions (flood fill from the border)."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return mask.astype(bool) | holes
