"""Procedural H&E-like whole-slide images with exact tumour annotations.

Purpose: exercise the *entire* pipeline (pyramidal TIFF reading, tissue
detection, tiling, training, stitching, quantification, UI) on a laptop or in CI
without downloading CAMELYON's 700+ GB. These slides are NOT a substitute for
real data; reported results must come from CAMELYON/PCam.

Look: glass background, pink stroma densely packed with small round dark nuclei
(lymphocytes, like a lymph node), and tumour regions with larger, paler,
irregular, more variable nuclei and more cytoplasm, mimicking metastatic
carcinoma. Written as a tiled, JPEG-compressed, multi-resolution BigTIFF with
TIFF resolution tags, i.e. the same container as CAMELYON16.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pathovision.wsi.annotations import Polygon, mask_to_polygons, rasterize, save_asap_xml

DESIGN_DS = 32  # level-0 px per design-mask px


def _smooth_noise(shape, sigma, rng):
    n = rng.standard_normal(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return (n - n.mean()) / (n.std() + 1e-6)


def design_masks(width: int, height: int, n_tumors: int, seed: int) -> Tuple[np.ndarray, List[Polygon]]:
    """Low-res tissue mask and level-0 tumour polygons."""
    rng = np.random.default_rng(seed)
    h, w = int(math.ceil(height / DESIGN_DS)), int(math.ceil(width / DESIGN_DS))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 4))):  # 2-3 tissue sections
        cx, cy = rng.uniform(0.25, 0.75) * w, rng.uniform(0.25, 0.75) * h
        rx, ry = rng.uniform(0.18, 0.32) * w, rng.uniform(0.18, 0.32) * h
        field = np.maximum(field, 1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2)
    field += 0.25 * _smooth_noise((h, w), max(h, w) / 40, rng)
    tissue = field > 0.1

    tumour = np.zeros((h, w), bool)
    ys, xs = np.nonzero(tissue)
    for i in range(n_tumors):
        k = int(rng.integers(0, len(xs)))
        cx, cy = xs[k], ys[k]
        # mix of sizes: some macro- (> 2 mm), some micrometastases
        r = rng.uniform(8, 18) if i % 3 == 2 else rng.uniform(25, 70)
        blob = 1 - ((xx - cx) / r) ** 2 - ((yy - cy) / (r * rng.uniform(0.6, 1.0))) ** 2
        blob += 0.35 * _smooth_noise((h, w), max(2.0, r / 4), rng)
        tumour |= blob > 0
    tumour &= tissue
    polys = mask_to_polygons(tumour, DESIGN_DS, min_area_px=3, simplify_px=0.5)
    return tissue, polys


class SlideRenderer:
    def __init__(self, width, height, mpp, tissue_lr, polys, seed):
        self.W, self.H, self.mpp = width, height, mpp
        self.tissue_lr = cv2.GaussianBlur(tissue_lr.astype(np.float32), (0, 0), 1.2)  # smooth section edges
        self.polys = polys
        self.seed = seed

    def render(self, x0: int, y0: int, size: int, ds: float) -> np.ndarray:
        """size x size RGB tile whose pixels are ``ds`` level-0 px; (x0, y0) level-0 top-left."""
        rng = np.random.default_rng((self.seed * 1_000_003 + int(x0) * 7919 + int(y0) * 104729 + int(ds)) % 2 ** 63)
        # tissue (smooth edge) and tumour (exact polygons) at tile resolution
        gx = (x0 + (np.arange(size) + 0.5) * ds) / DESIGN_DS - 0.5
        gy = (y0 + (np.arange(size) + 0.5) * ds) / DESIGN_DS - 0.5
        mx, my = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
        tis = cv2.remap(self.tissue_lr, mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.5
        tum = rasterize(self.polys, (size, size), ds, (x0, y0)) & tis

        img = np.empty((size, size, 3), np.float32)
        img[:] = (242, 241, 244)
        stroma = np.array([236, 172, 205], np.float32)
        tumour_bg = np.array([226, 160, 200], np.float32)
        tex = _smooth_noise((size, size), max(1.0, 2.0 / ds * 4), rng)[..., None]
        img[tis] = stroma + 10 * tex[tis]
        img[tum] = tumour_bg + 10 * tex[tum]
        if tis.any():
            um_per_px = self.mpp * ds
            area_mm2 = (size * um_per_px / 1000.0) ** 2
            if um_per_px <= 2.5:
                self._draw_nuclei(img, tis, tum, um_per_px, area_mm2, rng)
            else:  # nuclei below pixel size: render as tint proportional to nuclear density
                purple_n = np.array([95, 50, 140], np.float32)
                purple_t = np.array([140, 85, 175], np.float32)
                fn = np.clip(0.45 + 0.08 * _smooth_noise((size, size), 1.0, rng), 0, 1)[..., None]
                ft = np.clip(0.40 + 0.08 * _smooth_noise((size, size), 1.0, rng), 0, 1)[..., None]
                n_only = tis & ~tum
                img[n_only] = (1 - fn[n_only]) * img[n_only] + fn[n_only] * purple_n
                img[tum] = (1 - ft[tum]) * img[tum] + ft[tum] * purple_t
        img += rng.normal(0, 3, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_nuclei(img, tis, tum, um_per_px, area_mm2, rng):
        size = img.shape[0]
        # lymphocytes: ~6 um diameter, very dense; tumour nuclei: 10-16 um, pleomorphic
        for kind, density in (("lym", 9000), ("tum", 2600)):
            n = rng.poisson(density * area_mm2)
            if n == 0:
                continue
            pts = rng.uniform(0, size, (n, 2))
            ix = np.clip(pts.astype(int), 0, size - 1)
            region = tum if kind == "tum" else (tis & ~tum)
            pts = pts[region[ix[:, 1], ix[:, 0]]]
            for (x, y) in pts:
                if kind == "lym":
                    r = rng.uniform(2.6, 3.4) / um_per_px
                    axes, col = (r, r * rng.uniform(0.85, 1.0)), (rng.uniform(60, 95), rng.uniform(30, 55), rng.uniform(110, 150))
                else:
                    r = rng.uniform(5.0, 8.0) / um_per_px
                    axes, col = (r, r * rng.uniform(0.5, 0.85)), (rng.uniform(120, 160), rng.uniform(70, 105), rng.uniform(160, 195))
                cv2.ellipse(img, (int(x * 16), int(y * 16)),
                            (max(1, int(axes[0] * 16)), max(1, int(axes[1] * 16))),
                            float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA, 4)
                if kind == "tum" and axes[0] >= 3:  # nucleolus
                    cv2.circle(img, (int(x), int(y)), max(1, int(axes[0] * 0.18)), (80, 30, 110), -1)


def make_synthetic_slide(path: str, xml_path: Optional[str] = None, width: int = 40000, height: int = 30000,
                         mpp: float = 0.243, n_tumors: int = 5, seed: int = 0, tile: int = 256,
                         n_levels: int = 7, compression: str = "jpeg", verbose: bool = True) -> dict:
    """Write a pyramidal tiled BigTIFF (levels at 1, 2, 4, ... downsample) and its ASAP XML."""
    import tifffile

    tissue_lr, polys = design_masks(width, height, n_tumors, seed)
    if xml_path:
        save_asap_xml(polys, xml_path)
    rend = SlideRenderer(width, height, mpp, tissue_lr, polys, seed)
    res = 1e4 / mpp  # pixels per centimetre

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for lvl in range(n_levels):
            ds = 2 ** lvl
            lw, lh = int(math.ceil(width / ds)), int(math.ceil(height / ds))
            if min(lw, lh) < tile and lvl > 0:
                break
            nty, ntx = int(math.ceil(lh / tile)), int(math.ceil(lw / tile))

            def tiles(ds=ds, nty=nty, ntx=ntx):
                for ty in range(nty):
                    for tx in range(ntx):
                        yield rend.render(tx * tile * ds, ty * tile * ds, tile, ds)

            kw = dict(shape=(lh, lw, 3), dtype=np.uint8, tile=(tile, tile), photometric="rgb",
                      resolution=(res / ds, res / ds), resolutionunit="CENTIMETER",
                      subfiletype=0 if lvl == 0 else 1, metadata=None)
            if compression == "jpeg":
                kw["compression"] = "jpeg"
                kw["compressionargs"] = {"level": 90}
            elif compression:
                kw["compression"] = compression
            tw.write(tiles(), **kw)
            if verbose:
                print(f"  level {lvl}: {lw} x {lh} ({nty * ntx} tiles)", flush=True)
    return {"path": path, "xml": xml_path, "width": width, "height": height, "mpp": mpp,
            "n_tumor_polygons": sum(not p.is_exclusion for p in polys)}
