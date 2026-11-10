"""Unified, memory-safe access to whole-slide images.

A 40x CAMELYON slide is roughly 100,000 x 200,000 px. Decoded as RGB uint8 that is
~60 GB, so the full image is never materialised. Everything goes through
``read_region`` on the slide's image pyramid: we ask for a small window at the
pyramid level closest to the resolution we need, and resize only that window.

All public coordinates are *level-0* pixel coordinates (OpenSlide convention).
Resolutions are expressed in microns-per-pixel (mpp), which is what models are
actually sensitive to (a 10x model sees ~1.0 um/px regardless of scanner).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:  # OpenSlide is optional so that plain PNG/JPEG uploads still work.
    import openslide

    _HAS_OPENSLIDE = True
except Exception:  # pragma: no cover - depends on system libs
    openslide = None
    _HAS_OPENSLIDE = False

PYRAMID_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide"}
DEFAULT_MPP = 0.243  # CAMELYON16 level-0 resolution (40x objective)


@dataclass
class SlideInfo:
    path: str
    width: int
    height: int
    level_count: int
    level_dimensions: Tuple[Tuple[int, int], ...]
    level_downsamples: Tuple[float, ...]
    mpp: float
    mpp_source: str
    vendor: str

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    def to_dict(self) -> dict:
        return {
            "path": os.path.basename(self.path),
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.megapixels, 1),
            "level_count": self.level_count,
            "level_dimensions": [list(d) for d in self.level_dimensions],
            "level_downsamples": [round(d, 3) for d in self.level_downsamples],
            "mpp": round(self.mpp, 4),
            "mpp_source": self.mpp_source,
            "vendor": self.vendor,
        }


def _mpp_from_properties(props) -> Tuple[Optional[float], str]:
    """Find level-0 microns-per-pixel from OpenSlide properties."""
    for key in ("openslide.mpp-x", "aperio.MPP"):
        if key in props:
            try:
                v = float(props[key])
                if v > 0:
                    return v, key
            except ValueError:
                pass
    # Generic TIFF (CAMELYON16 is stored this way): use the TIFF resolution tags.
    xres = props.get("tiff.XResolution")
    unit = props.get("tiff.ResolutionUnit", "").lower()
    if xres:
        try:
            xres = float(xres)
            if xres > 0:
                if unit in ("centimeter", "3"):
                    return 1e4 / xres, "tiff.XResolution(cm)"
                if unit in ("inch", "2"):
                    return 25400.0 / xres, "tiff.XResolution(inch)"
        except ValueError:
            pass
    return None, "default"


class WSIReader:
    """Read regions of a whole-slide image at any resolution without loading it whole.

    Uses OpenSlide for pyramidal formats; falls back to an in-memory single-level
    image for ordinary PNG/JPEG files (only sensible for small images).
    """

    def __init__(self, path: str, default_mpp: float = DEFAULT_MPP, mpp_override: Optional[float] = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self._slide = None
        self._array: Optional[np.ndarray] = None
        ext = os.path.splitext(path)[1].lower()

        if _HAS_OPENSLIDE and ext in PYRAMID_EXTS:
            try:
                self._slide = openslide.OpenSlide(path)
            except Exception:
                self._slide = None

        if self._slide is not None:
            s = self._slide
            mpp, src = _mpp_from_properties(s.properties)
            vendor = s.properties.get("openslide.vendor", "unknown")
            dims = tuple(tuple(int(v) for v in d) for d in s.level_dimensions)
            downs = tuple(float(d) for d in s.level_downsamples)
        else:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as im:
                self._array = np.asarray(im.convert("RGB"))
            h, w = self._array.shape[:2]
            mpp, src, vendor = None, "default", "plain-image"
            dims, downs = ((w, h),), (1.0,)

        if mpp_override:
            mpp, src = float(mpp_override), "override"
        if mpp is None:
            mpp = default_mpp
        self.info = SlideInfo(
            path=path,
            width=dims[0][0],
            height=dims[0][1],
            level_count=len(dims),
            level_dimensions=dims,
            level_downsamples=downs,
            mpp=float(mpp),
            mpp_source=src,
            vendor=vendor,
        )

    # ------------------------------------------------------------------ basics
    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.info.width, self.info.height

    @property
    def mpp(self) -> float:
        return self.info.mpp

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None
        self._array = None

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- reading
    def best_level_for_downsample(self, downsample: float) -> int:
        """Highest-index (lowest-resolution) level whose downsample is <= requested.

        Reading from a level that is *finer* than needed and shrinking keeps image
        quality; reading a coarser level and upsampling would invent detail.
        """
        best = 0
        for i, d in enumerate(self.info.level_downsamples):
            if d <= downsample * 1.01:  # tolerate float noise (e.g. 3.9999)
                best = i
        return best

    def read_region(self, x: int, y: int, level: int, w: int, h: int) -> np.ndarray:
        """RGB uint8 array of size (h, w) at ``level``; (x, y) are level-0 coords.

        Regions extending past the slide border are padded with white (background).
        """
        if self._slide is not None:
            region = self._slide.read_region((int(x), int(y)), int(level), (int(w), int(h)))
            arr = np.asarray(region)  # RGBA, alpha=0 outside the scanned area
            rgb = arr[..., :3].copy()
            alpha = arr[..., 3]
            if alpha.min() < 255:
                rgb[alpha == 0] = 255
            return rgb
        # plain image fallback (single level)
        out = np.full((int(h), int(w), 3), 255, np.uint8)
        H, W = self._array.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
        if x1 > x0 and y1 > y0:
            out[y0 - int(y): y1 - int(y), x0 - int(x): x1 - int(x)] = self._array[y0:y1, x0:x1]
        return out

    def read_at_downsample(self, x: int, y: int, w_out: int, h_out: int, downsample: float) -> np.ndarray:
        """Read a (h_out, w_out) image whose pixels are ``downsample`` level-0 px wide."""
        level = self.best_level_for_downsample(downsample)
        level_ds = self.info.level_downsamples[level]
        scale = downsample / level_ds  # >= 1: shrink factor applied after reading
        w_lvl = max(1, int(round(w_out * scale)))
        h_lvl = max(1, int(round(h_out * scale)))
        img = self.read_region(x, y, level, w_lvl, h_lvl)
        if (w_lvl, h_lvl) != (w_out, h_out):
            interp = cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (w_out, h_out), interpolation=interp)
        return img

    def read_at_mpp(self, x: int, y: int, size: int, target_mpp: float) -> np.ndarray:
        """Square patch of ``size`` px at ``target_mpp`` microns/px, top-left at level-0 (x, y)."""
        return self.read_at_downsample(x, y, size, size, target_mpp / self.mpp)

    def footprint(self, size: int, target_mpp: float) -> float:
        """Width in level-0 pixels covered by a ``size``-px patch at ``target_mpp``."""
        return size * target_mpp / self.mpp

    def thumbnail(self, max_side: int = 2048) -> Tuple[np.ndarray, float]:
        """Whole-slide overview image and its downsample factor relative to level 0."""
        W, H = self.dimensions
        downsample = max(1.0, max(W, H) / float(max_side))
        w_out = max(1, int(math.floor(W / downsample)))
        h_out = max(1, int(math.floor(H / downsample)))
        if self._slide is not None:
            img = self.read_at_downsample(0, 0, w_out, h_out, downsample)
        else:
            img = cv2.resize(self._array, (w_out, h_out), interpolation=cv2.INTER_AREA)
        # exact downsample after integer rounding
        return img, W / float(w_out)
