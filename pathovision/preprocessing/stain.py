"""Stain normalisation and stain augmentation for H&E images.

Slides from different hospitals look different (stain batch, thickness, scanner).
A model trained on one centre's colours often fails on another's. Two remedies:

* **Macenko normalisation** (inference/pre-processing): convert RGB to optical
  density (Beer-Lambert: OD = -log(I / I0)), estimate the two stain vectors
  (haematoxylin, eosin) from the plane spanned by the top-2 eigenvectors of the
  tissue OD, then re-express the image with a reference stain matrix and
  reference concentrations.
* **HED jitter** (training augmentation): perturb the stain concentrations
  directly so the model learns to ignore colour variation.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Standard reference values (widely used, from the original Macenko implementation).
HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])


def rgb_to_od(img: np.ndarray, io: float = 240.0) -> np.ndarray:
    img = img.astype(np.float64)
    return -np.log((img + 1.0) / io)


def od_to_rgb(od: np.ndarray, io: float = 240.0) -> np.ndarray:
    return np.clip(io * np.exp(-od), 0, 255).astype(np.uint8)


def estimate_stain_matrix(img: np.ndarray, io: float = 240.0, beta: float = 0.15, alpha: float = 1.0,
                          min_pixels: int = 100) -> Optional[np.ndarray]:
    """3x2 stain matrix (columns = H, E unit vectors in OD space) or None if not enough tissue."""
    od = rgb_to_od(img.reshape(-1, 3), io)
    od_hat = od[np.all(od > beta, axis=1)]  # drop transparent/background pixels
    if od_hat.shape[0] < min_pixels:
        return None
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = eigvecs[:, 1:3]  # two largest eigenvalues
    proj = od_hat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi, max_phi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])
    # Eigenvector signs are arbitrary; stain OD vectors are physically positive.
    v1 = v1 if v1.sum() >= 0 else -v1
    v2 = v2 if v2.sum() >= 0 else -v2
    # Haematoxylin absorbs red light most strongly -> larger R component goes first.
    he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    return he / np.linalg.norm(he, axis=0, keepdims=True)


def stain_concentrations(img: np.ndarray, he: np.ndarray, io: float = 240.0) -> np.ndarray:
    od = rgb_to_od(img.reshape(-1, 3), io)
    c, *_ = np.linalg.lstsq(he, od.T, rcond=None)
    return c  # (2, N)


class MacenkoNormalizer:
    """Fit on a reference tile (or use literature defaults), then transform tiles."""

    def __init__(self, io: float = 240.0, beta: float = 0.15, alpha: float = 1.0):
        self.io, self.beta, self.alpha = io, beta, alpha
        self.he_ref = HE_REF.copy()
        self.max_c_ref = MAX_C_REF.copy()

    def fit(self, target: np.ndarray) -> "MacenkoNormalizer":
        he = estimate_stain_matrix(target, self.io, self.beta, self.alpha)
        if he is None:
            raise ValueError("Reference image contains too little tissue to estimate stains")
        c = stain_concentrations(target, he, self.io)
        self.he_ref = he
        self.max_c_ref = np.percentile(c, 99, axis=1)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalised RGB uint8 image; returns the input unchanged if it is (almost) all background."""
        h, w, _ = img.shape
        he = estimate_stain_matrix(img, self.io, self.beta, self.alpha)
        if he is None:
            return img
        c = stain_concentrations(img, he, self.io)
        max_c = np.percentile(c, 99, axis=1)
        max_c = np.where(max_c < 1e-6, 1.0, max_c)
        c = c * (self.max_c_ref / max_c)[:, None]
        od = (self.he_ref @ c).T
        return od_to_rgb(od, self.io).reshape(h, w, 3)

    def state_dict(self) -> dict:
        return {"he_ref": self.he_ref.tolist(), "max_c_ref": self.max_c_ref.tolist(), "io": self.io}

    @classmethod
    def from_state_dict(cls, d: dict) -> "MacenkoNormalizer":
        n = cls(io=d.get("io", 240.0))
        n.he_ref = np.array(d["he_ref"])
        n.max_c_ref = np.array(d["max_c_ref"])
        return n


# Fixed RGB->HED matrix (Ruifrok & Johnston), same as skimage.color.rgb2hed.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_RGB_FROM_HED[2] = np.cross(_RGB_FROM_HED[0], _RGB_FROM_HED[1])
_RGB_FROM_HED = _RGB_FROM_HED / np.linalg.norm(_RGB_FROM_HED, axis=1, keepdims=True)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def hed_jitter(img: np.ndarray, sigma: float = 0.05, bias: float = 0.05,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random per-stain scale/shift in HED space (Tellez et al., 2018). uint8 in, uint8 out."""
    rng = rng or np.random.default_rng()
    x = np.maximum(img.astype(np.float64), 1.0) / 255.0
    hed = (-np.log(x)).reshape(-1, 3) @ _HED_FROM_RGB
    a = rng.uniform(1 - sigma, 1 + sigma, size=3)
    b = rng.uniform(-bias, bias, size=3)
    hed = hed * a + b
    rgb = np.exp(-(hed @ _RGB_FROM_HED))
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8).reshape(img.shape)
