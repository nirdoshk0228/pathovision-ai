"""Turn a tumour probability map into clinically meaningful numbers.

Per slide: tissue area, tumour area and burden (% of tissue), number of lesions.
Per lesion: area (mm^2), largest diameter (mm), centroid, mean/max probability.

The largest diameter drives the size category used for sentinel lymph-node
metastases (AJCC/CAMELYON17 convention):

* isolated tumour cells (ITC): <= 0.2 mm
* micrometastasis:            > 0.2 mm and <= 2.0 mm
* macrometastasis:            > 2.0 mm

Largest diameter = maximum Feret diameter: the largest distance between any two
points of the lesion. Any farthest pair lies on the convex hull, so we compute
the hull first (O(n log n)) and then compare only hull vertices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class Lesion:
    id: int
    area_mm2: float
    max_diameter_mm: float
    centroid_x: float  # level-0 px
    centroid_y: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    mean_prob: float
    max_prob: float
    category: str


def size_category(diameter_mm: float) -> str:
    if diameter_mm > 2.0:
        return "macrometastasis"
    if diameter_mm > 0.2:
        return "micrometastasis"
    return "ITC"


def max_feret_diameter(points: np.ndarray) -> float:
    """Largest pairwise distance between points (pixel units)."""
    if len(points) < 2:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) > 400:  # keep the O(h^2) step bounded for very ragged hulls
        hull = hull[np.linspace(0, len(hull) - 1, 400).astype(int)]
    d = hull[:, None, :] - hull[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def quantify(
    prob: np.ndarray,
    tissue: np.ndarray,
    downsample: float,
    mpp: float,
    threshold: float = 0.5,
    min_lesion_um2: float = 200.0,
) -> Dict:
    """Quantify a probability map living at ``downsample`` level-0 px per pixel.

    ``tissue`` must have the same shape as ``prob``. Connected components smaller
    than ``min_lesion_um2`` (default 200 um^2, roughly a small cluster of tumour
    cells) are treated as noise and not reported as lesions.
    """
    px_um = downsample * mpp  # microns per map pixel
    px_mm2 = (px_um / 1000.0) ** 2
    min_lesion_px = max(1, int(np.ceil(min_lesion_um2 / (px_um ** 2))))
    tumour = (prob >= threshold) & tissue
    tissue_mm2 = float(tissue.sum()) * px_mm2
    n, labels, stats, cents = cv2.connectedComponentsWithStats(tumour.astype(np.uint8), connectivity=8)
    lesions: List[Lesion] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_lesion_px:
            continue
        ys, xs = np.nonzero(labels == i)
        # pixel corners make the diameter of tiny lesions realistic (a 1-px lesion is 1 px wide, not 0)
        pts = np.concatenate([np.stack([xs, ys], 1), np.stack([xs + 1, ys + 1], 1)])
        diam_mm = max_feret_diameter(pts) * px_um / 1000.0
        p = prob[ys, xs]
        x, y, w, h = (stats[i, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        lesions.append(Lesion(
            id=len(lesions) + 1, area_mm2=area * px_mm2, max_diameter_mm=diam_mm,
            centroid_x=float((cents[i][0] + 0.5) * downsample), centroid_y=float((cents[i][1] + 0.5) * downsample),
            bbox_x=float(x * downsample), bbox_y=float(y * downsample), bbox_w=float(w * downsample),
            bbox_h=float(h * downsample), mean_prob=float(p.mean()), max_prob=float(p.max()),
            category=size_category(diam_mm)))
    lesions.sort(key=lambda l: l.area_mm2, reverse=True)
    for k, l in enumerate(lesions, 1):
        l.id = k
    tumour_mm2 = float(sum(l.area_mm2 for l in lesions))
    largest = max((l.max_diameter_mm for l in lesions), default=0.0)
    return {
        "threshold": threshold,
        "min_lesion_um2": min_lesion_um2,
        "map_downsample": downsample,
        "map_um_per_px": px_um,
        "tissue_area_mm2": tissue_mm2,
        "tumor_area_mm2": tumour_mm2,
        "tumor_burden_pct": 100.0 * tumour_mm2 / tissue_mm2 if tissue_mm2 > 0 else 0.0,
        "n_lesions": len(lesions),
        "largest_lesion_diameter_mm": largest,
        "slide_category": size_category(largest) if lesions else "negative",
        "lesions": [asdict(l) for l in lesions],
    }
