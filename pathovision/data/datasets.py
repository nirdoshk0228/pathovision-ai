"""PyTorch datasets for training (PCam, Kaggle HCD, segmentation patches) and slide inference."""
from __future__ import annotations

import glob
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pathovision.utils.common import to_tensor


class PCamH5Dataset(Dataset):
    """PatchCamelyon HDF5 files (x: N x 96 x 96 x 3 uint8, y: N x 1 x 1 x 1).

    Our own ``scripts/extract_patches.py`` writes the same format, so CAMELYON
    patches at any resolution can be used interchangeably with PCam.

    h5py handles are not fork-safe, so each DataLoader worker opens its own
    handle lazily on first access.
    """

    def __init__(self, x_path: str, y_path: str, augment: Optional[Callable] = None,
                 indices: Optional[Sequence[int]] = None):
        import h5py

        self.x_path, self.y_path, self.augment = x_path, y_path, augment
        with h5py.File(y_path, "r") as f:
            self.labels = f["y"][:].reshape(-1).astype(np.float32)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self._x = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._x is None:
            import h5py

            self._x = h5py.File(self.x_path, "r")["x"]
        j = int(self.indices[i])
        img = self._x[j]
        if self.augment is not None:
            img, _ = self.augment(img)
        return to_tensor(img)[0], torch.tensor(self.labels[j])


class KaggleHCDDataset(Dataset):
    """Kaggle 'Histopathologic Cancer Detection' layout: train/<id>.tif + train_labels.csv."""

    def __init__(self, img_dir: str, ids: Sequence[str], labels: Sequence[int], augment: Optional[Callable] = None):
        self.img_dir, self.ids, self.labels, self.augment = img_dir, list(ids), np.asarray(labels, np.float32), augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, self.ids[i] + ".tif")), cv2.COLOR_BGR2RGB)
        if self.augment is not None:
            img, _ = self.augment(img)
        return to_tensor(img)[0], torch.tensor(self.labels[i])


class SegPatchDataset(Dataset):
    """Folder of paired patches: <root>/images/*.png and <root>/masks/*.png (0/255)."""

    def __init__(self, root: str, augment: Optional[Callable] = None, files: Optional[List[str]] = None):
        self.root, self.augment = root, augment
        self.files = files if files is not None else sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(root, "images", "*.png")))
        if not self.files:
            raise FileNotFoundError(f"no patches under {root}/images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        name = self.files[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, "masks", name), cv2.IMREAD_GRAYSCALE) > 127
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return to_tensor(img)[0], torch.from_numpy(mask.astype(np.float32))[None]


class WSITileDataset(Dataset):
    """Reads patches straight from the slide for batched inference.

    Each worker process opens its own reader (OpenSlide handles cannot be shared
    across processes), so tile decoding runs in parallel with GPU compute.
    """

    def __init__(self, slide_path: str, coords: np.ndarray, patch_size: int, target_mpp: float,
                 normalizer=None, mpp_override: Optional[float] = None):
        self.slide_path, self.coords = slide_path, coords
        self.patch_size, self.target_mpp = patch_size, target_mpp
        self.normalizer, self.mpp_override = normalizer, mpp_override
        self._reader = None

    def __len__(self) -> int:
        return len(self.coords)

    def _get_reader(self):
        if self._reader is None:
            from pathovision.wsi.reader import WSIReader

            self._reader = WSIReader(self.slide_path, mpp_override=self.mpp_override)
        return self._reader

    def __getitem__(self, i: int):
        x, y = int(self.coords[i, 0]), int(self.coords[i, 1])
        img = self._get_reader().read_at_mpp(x, y, self.patch_size, self.target_mpp)
        if self.normalizer is not None:
            img = self.normalizer.transform(img)
        return to_tensor(img)[0], i
