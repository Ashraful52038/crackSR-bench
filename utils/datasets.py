"""PyTorch datasets for super-resolution and segmentation.

All datasets return tensors in [0, 1]; normalisation happens inside the models
so that SR outputs can be fed to the segmentation model directly.
Classification uses torchvision's ImageFolder (see engine/cls.py).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, degrade
from .image_io import img2tensor, list_images, mod_crop, pair_files, read_mask, read_rgb


def _rng() -> np.random.Generator:
    # torch seeds every DataLoader worker differently, so this is reproducible
    # under utils.common.set_seed and still different per worker.
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))


def _random_crop(arrays: list[np.ndarray], size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Same random crop for all arrays; pads (reflect for images, zero for masks) if too small."""
    h, w = arrays[0].shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        padded = []
        for a in arrays:
            border = cv2.BORDER_REFLECT_101 if a.ndim == 3 else cv2.BORDER_CONSTANT
            padded.append(cv2.copyMakeBorder(a, 0, ph, 0, pw, border, value=0))
        arrays = padded
        h, w = arrays[0].shape[:2]
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return [a[top: top + size, left: left + size] for a in arrays]


def _augment(arrays: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """Same flips / 90-degree rotation for all arrays."""
    if rng.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[::-1] for a in arrays]
    k = int(rng.integers(4))
    arrays = [np.rot90(a, k) for a in arrays]
    return [np.ascontiguousarray(a) for a in arrays]


class SRTrainDataset(Dataset):
    """HR images only; LR patches are synthesised on the fly with `degradation`."""

    def __init__(self, hr_dir, scale: int, patch_size: int, degradation=None, repeat: int = 1):
        self.files = list_images(hr_dir)
        if not self.files:
            raise FileNotFoundError(f"No images in {hr_dir}")
        self.scale = scale
        self.patch = patch_size - patch_size % scale
        self.deg = DegradationConfig.from_any(degradation, scale)
        self.repeat = repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        rng = _rng()
        hr = read_rgb(self.files[idx % len(self.files)])
        hr = _random_crop([hr], self.patch, rng)[0]
        hr = _augment([hr], rng)[0]
        lr = degrade(hr, self.deg, rng)
        return {"lr": img2tensor(lr), "hr": img2tensor(hr)}


class SRPairedDataset(Dataset):
    """Fixed LR/HR pairs (LR made once by scripts/degrade.py) for validation and testing."""

    def __init__(self, lr_dir, hr_dir, scale: int, limit: int | None = None):
        self.hr_files = list_images(hr_dir)[:limit]
        if not self.hr_files:
            raise FileNotFoundError(f"No images in {hr_dir}")
        self.lr_files = pair_files(self.hr_files, lr_dir)
        self.scale = scale

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        lr = read_rgb(self.lr_files[idx])
        hr = mod_crop(read_rgb(self.hr_files[idx]), self.scale)
        h, w = lr.shape[0] * self.scale, lr.shape[1] * self.scale
        if hr.shape[0] < h or hr.shape[1] < w:
            raise ValueError(f"LR {self.lr_files[idx].name} is too large for HR at x{self.scale}")
        hr = hr[:h, :w]
        return {"lr": img2tensor(lr), "hr": img2tensor(hr), "name": self.hr_files[idx].stem}


class SegDataset(Dataset):
    """Image + binary crack mask pairs. Training mode returns random crops."""

    def __init__(self, images_dir, masks_dir, crop_size: int | None = None, augment: bool = False):
        self.images = list_images(images_dir)
        if not self.images:
            raise FileNotFoundError(f"No images in {images_dir}")
        self.masks = pair_files(self.images, masks_dir)
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = read_rgb(self.images[idx])
        mask = read_mask(self.masks[idx])
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.crop_size or self.augment:
            rng = _rng()
            if self.crop_size:
                img, mask = _random_crop([img, mask], self.crop_size, rng)
            if self.augment:
                img, mask = _augment([img, mask], rng)
        return {
            "image": img2tensor(img),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).float()[None],
            "name": Path(self.images[idx]).stem,
        }
