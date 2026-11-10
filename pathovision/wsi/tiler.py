"""Turn a gigapixel slide into a grid of model-sized patches.

The grid is defined in level-0 coordinates. A patch of ``patch_size`` pixels at
``target_mpp`` covers ``footprint = patch_size * target_mpp / slide_mpp`` level-0
pixels. With CAMELYON (0.243 um/px) and a 10x model (0.972 um/px, 96 px) every
patch covers 384 x 384 level-0 pixels.

Which grid cells contain tissue is answered with a *summed-area table* (integral
image) of the low-resolution tissue mask: after one O(N) pass, the tissue fraction
inside any rectangle is four lookups, so filtering tens of thousands of candidate
tiles is O(1) each instead of O(area).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int  # level-0 top-left
    y: int
    row: int  # grid index
    col: int
    tissue_frac: float


class IntegralImage:
    """Summed-area table supporting O(1) rectangle sums."""

    def __init__(self, mask: np.ndarray):
        m = mask.astype(np.float64)
        self.sat = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float64)
        self.sat[1:, 1:] = m.cumsum(0).cumsum(1)
        self.h, self.w = m.shape

    def rect_mean(self, x0, y0, x1, y1):
        """Mean of mask over [x0, x1) x [y0, y1); arrays allowed (vectorised)."""
        x0 = np.clip(x0, 0, self.w)
        x1 = np.clip(x1, 0, self.w)
        y0 = np.clip(y0, 0, self.h)
        y1 = np.clip(y1, 0, self.h)
        s = self.sat
        total = s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]
        area = np.maximum((x1 - x0) * (y1 - y0), 1)
        return total / area


class TileGrid:
    """Regular grid of patches over the slide, optionally restricted to tissue.

    Args:
        slide_wh: level-0 (width, height).
        footprint: level-0 size of one patch (float allowed).
        overlap: fraction of overlap between neighbouring patches (0 <= overlap < 1).
        tissue: boolean tissue mask from the thumbnail.
        mask_downsample: level-0 pixels per tissue-mask pixel.
        min_tissue: minimum tissue fraction for a patch to be kept.
    """

    def __init__(
        self,
        slide_wh: Tuple[int, int],
        footprint: float,
        overlap: float = 0.0,
        tissue: Optional[np.ndarray] = None,
        mask_downsample: float = 1.0,
        min_tissue: float = 0.25,
    ):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.W, self.H = slide_wh
        self.footprint = float(footprint)
        self.stride = self.footprint * (1.0 - overlap)
        self.n_cols = max(1, int(math.ceil((self.W - self.footprint) / self.stride)) + 1)
        self.n_rows = max(1, int(math.ceil((self.H - self.footprint) / self.stride)) + 1)

        rows, cols = np.mgrid[0: self.n_rows, 0: self.n_cols]
        xs = np.round(cols * self.stride).astype(np.int64)
        ys = np.round(rows * self.stride).astype(np.int64)

        if tissue is not None:
            ii = IntegralImage(tissue)
            ds = float(mask_downsample)
            mx0 = np.floor(xs / ds).astype(np.int64)
            my0 = np.floor(ys / ds).astype(np.int64)
            mx1 = np.maximum(mx0 + 1, np.ceil((xs + self.footprint) / ds).astype(np.int64))
            my1 = np.maximum(my0 + 1, np.ceil((ys + self.footprint) / ds).astype(np.int64))
            frac = ii.rect_mean(mx0, my0, mx1, my1)
        else:
            frac = np.ones_like(xs, dtype=np.float64)

        keep = frac >= min_tissue
        self._x = xs[keep]
        self._y = ys[keep]
        self._r = rows[keep]
        self._c = cols[keep]
        self._f = frac[keep]
        self.total_cells = int(xs.size)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self.n_rows, self.n_cols

    def __len__(self) -> int:
        return int(self._x.size)

    def __iter__(self) -> Iterator[Tile]:
        for x, y, r, c, f in zip(self._x, self._y, self._r, self._c, self._f):
            yield Tile(int(x), int(y), int(r), int(c), float(f))

    def coords(self) -> np.ndarray:
        """(N, 4) int array of [x, y, row, col]."""
        return np.stack([self._x, self._y, self._r, self._c], axis=1)

    @property
    def tissue_fracs(self) -> np.ndarray:
        return self._f
