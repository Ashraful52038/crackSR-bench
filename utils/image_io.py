"""Image reading/writing and conversions. Images are RGB uint8 HxWx3 numpy arrays;
tensors are float in [0, 1] with shape (3, H, W)."""
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = ("_mask", "_label", "_gt", "_seg")


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def file_key(path: Path) -> str:
    """Stem with common mask suffixes removed, so `001.jpg` matches `001_mask.png`."""
    stem = path.stem
    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_folder(folder: str | Path) -> dict[str, Path]:
    return {file_key(p): p for p in list_images(folder)}


def pair_files(files: list[Path], folder: str | Path) -> list[Path]:
    """For each file, find the file in `folder` with the same key. Raises if any is missing."""
    index = index_folder(folder)
    missing = [p.name for p in files if file_key(p) not in index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} file(s) have no match in {folder}, e.g. {missing[:5]}"
        )
    return [index[file_key(p)] for p in files]


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Binary crack mask as uint8 {0, 1}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask: {path}")
    return (mask > threshold).astype(np.uint8)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def img2tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div(255.0)


def tensor2img(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
