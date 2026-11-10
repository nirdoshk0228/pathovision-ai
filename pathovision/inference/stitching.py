"""Blend overlapping tile predictions into one slide-level probability map.

Problem: a segmentation CNN is least reliable near patch borders (less context,
zero padding). If tiles are simply pasted side by side, the output shows visible
seams and lesions crossing tile edges get cut.

Solution: predict on *overlapping* tiles and blend with a weight window that is
1 in the centre and falls towards the borders. Each output pixel becomes a
weighted average of every tile prediction that covers it:

    P(x) = sum_t w_t(x) * p_t(x) / sum_t w_t(x)

Memory: the canvas lives at a reduced resolution (``downsample`` level-0 pixels
per canvas pixel). At 32x a 100k x 200k slide becomes 3125 x 6250, two float32
buffers = ~156 MB, instead of the ~80 GB a level-0 float mask would need.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def weight_window(size: int, kind: str = "hann", floor: float = 0.05) -> np.ndarray:
    """2-D blending weights, max 1 at the centre, ``floor`` at the edges (never 0 -> no holes)."""
    if kind == "flat":
        return np.ones((size, size), np.float32)
    w1 = np.hanning(size + 2)[1:-1] if kind == "hann" else np.ones(size)
    w = np.outer(w1, w1).astype(np.float32)
    return floor + (1 - floor) * w / w.max()


class StitchAccumulator:
    def __init__(self, slide_wh: Tuple[int, int], downsample: float, window: str = "hann"):
        W, H = slide_wh
        self.ds = float(downsample)
        self.shape = (int(np.ceil(H / self.ds)), int(np.ceil(W / self.ds)))
        self.acc = np.zeros(self.shape, np.float32)
        self.wsum = np.zeros(self.shape, np.float32)
        self.window = window
        self._win_cache = {}

    @property
    def nbytes(self) -> int:
        return self.acc.nbytes + self.wsum.nbytes

    def _win(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._win_cache:
            if self.window == "flat":
                self._win_cache[key] = np.ones((h, w), np.float32)
            else:
                n = max(h, w)
                self._win_cache[key] = cv2.resize(weight_window(n, self.window), (w, h), interpolation=cv2.INTER_LINEAR)
        return self._win_cache[key]

    def add(self, prob: np.ndarray, x0: float, y0: float, footprint: float) -> None:
        """Add a square tile prediction ``prob`` covering level-0 [x0, x0+footprint)^2.

        Start *and* end edges are rounded from level-0 coordinates (rather than
        rounding the start and adding a rounded size), so tiles that touch in
        level-0 space also touch on the canvas: no 1-px gaps, no double counting.
        """
        r0 = int(round(y0 / self.ds))
        c0 = int(round(x0 / self.ds))
        r1 = max(r0 + 1, int(round((y0 + footprint) / self.ds)))
        c1 = max(c0 + 1, int(round((x0 + footprint) / self.ds)))
        th, tw = r1 - r0, c1 - c0
        interp = cv2.INTER_AREA if prob.shape[0] > th else cv2.INTER_LINEAR
        tile = cv2.resize(prob.astype(np.float32), (tw, th), interpolation=interp)
        win = self._win(th, tw)
        # clip to canvas
        cr0, cc0 = max(0, r0), max(0, c0)
        cr1, cc1 = min(self.shape[0], r1), min(self.shape[1], c1)
        if cr1 <= cr0 or cc1 <= cc0:
            return
        sl_t = (slice(cr0 - r0, cr1 - r0), slice(cc0 - c0, cc1 - c0))
        self.acc[cr0:cr1, cc0:cc1] += tile[sl_t] * win[sl_t]
        self.wsum[cr0:cr1, cc0:cc1] += win[sl_t]

    def coverage(self) -> np.ndarray:
        return self.wsum > 0

    def result(self) -> np.ndarray:
        out = np.zeros(self.shape, np.float32)
        m = self.wsum > 0
        out[m] = self.acc[m] / self.wsum[m]
        return out
