"""Low-resolution image synthesis.

`bicubic` mode is the classic setting used by most SR papers (and by the ESPCNN
baseline paper). `real` mode follows the second-order degradation model of
Real-ESRGAN (Wang et al., ICCVW 2021): blur -> resize -> noise -> JPEG, applied
twice, which is much closer to UAV / phone inspection images.

Implemented with numpy + OpenCV so it runs in DataLoader workers and in
scripts/degrade.py without a GPU. The sinc filter of Real-ESRGAN is omitted.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from .config import load_yaml

_INTERP = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)


@dataclass
class DegradationConfig:
    scale: int = 4
    mode: str = "real"  # "bicubic" | "real"
    # first degradation stage
    blur_prob: float = 1.0
    kernel_range: tuple = (7, 21)
    sigma_range: tuple = (0.2, 3.0)
    resize_prob: tuple = (0.2, 0.7, 0.1)  # up, down, keep
    resize_range: tuple = (0.15, 1.5)
    gaussian_noise_prob: float = 0.5
    noise_range: tuple = (1.0, 30.0)
    poisson_scale_range: tuple = (0.05, 3.0)
    gray_noise_prob: float = 0.4
    jpeg_range: tuple = (30, 95)
    # second degradation stage
    second_blur_prob: float = 0.8
    sigma_range2: tuple = (0.2, 1.5)
    resize_prob2: tuple = (0.3, 0.4, 0.3)
    resize_range2: tuple = (0.3, 1.2)
    gaussian_noise_prob2: float = 0.5
    noise_range2: tuple = (1.0, 25.0)
    poisson_scale_range2: tuple = (0.05, 2.5)
    gray_noise_prob2: float = 0.4
    jpeg_range2: tuple = (30, 95)

    @classmethod
    def from_any(cls, spec: "dict | str | Path | DegradationConfig | None", scale: int) -> "DegradationConfig":
        """Build from a dict, a YAML path, or None (bicubic). `scale` always wins."""
        if isinstance(spec, DegradationConfig):
            values = {f.name: getattr(spec, f.name) for f in fields(cls)}
        elif spec is None:
            values = {"mode": "bicubic"}
        elif isinstance(spec, (str, Path)):
            values = load_yaml(spec)
        else:
            values = dict(spec)
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown degradation option(s): {sorted(unknown)}")
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        values["scale"] = scale
        cfg = cls(**values)
        if cfg.mode not in ("bicubic", "real"):
            raise ValueError(f"Degradation mode must be 'bicubic' or 'real', got {cfg.mode!r}")
        return cfg


def gaussian_kernel(ksize: int, sigma_x: float, sigma_y: float, theta: float) -> np.ndarray:
    ax = np.arange(ksize) - ksize // 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    k = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
    return (k / k.sum()).astype(np.float32)


def _random_blur(img, rng, kernel_range, sigma_range):
    ksize = int(rng.integers(kernel_range[0] // 2, kernel_range[1] // 2 + 1)) * 2 + 1
    sx = rng.uniform(*sigma_range)
    if rng.random() < 0.5:  # isotropic
        sy, theta = sx, 0.0
    else:  # anisotropic
        sy, theta = rng.uniform(*sigma_range), rng.uniform(0, np.pi)
    return cv2.filter2D(img, -1, gaussian_kernel(ksize, sx, sy, theta), borderType=cv2.BORDER_REFLECT_101)


def _sample_scale(rng, probs, scale_range) -> float:
    mode = rng.choice(3, p=np.asarray(probs) / np.sum(probs))
    if mode == 0:
        return rng.uniform(1.0, scale_range[1])
    if mode == 1:
        return rng.uniform(scale_range[0], 1.0)
    return 1.0


def _resize(img, rng, width, height):
    return cv2.resize(img, (max(1, int(width)), max(1, int(height))), interpolation=_INTERP[rng.integers(3)])


def _gaussian_noise(img, rng, sigma_range, gray_prob):
    sigma = rng.uniform(*sigma_range) / 255.0
    shape = img.shape[:2] + (1,) if rng.random() < gray_prob else img.shape
    return img + rng.normal(0, sigma, shape).astype(np.float32)


def _poisson_noise(img, rng, scale_range, gray_prob):
    scale = rng.uniform(*scale_range)
    src = np.clip(img, 0, 1)
    if rng.random() < gray_prob:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)[..., None]
    n_levels = max(len(np.unique(np.round(src * 255))), 2)
    vals = 2 ** np.ceil(np.log2(n_levels))
    noise = rng.poisson(src * vals) / vals - src
    return img + (noise * scale).astype(np.float32)


def _noise(img, rng, gaussian_prob, noise_range, poisson_range, gray_prob):
    if rng.random() < gaussian_prob:
        return _gaussian_noise(img, rng, noise_range, gray_prob)
    return _poisson_noise(img, rng, poisson_range, gray_prob)


def _jpeg(img, rng, quality_range):
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def degrade(hr: np.ndarray, cfg: DegradationConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    """Turn an RGB uint8 HR image into an LR image of size (H // scale, W // scale)."""
    rng = rng if rng is not None else np.random.default_rng()
    h, w = hr.shape[:2]
    tw, th = w // cfg.scale, h // cfg.scale
    if cfg.mode == "bicubic":
        return cv2.resize(hr, (tw, th), interpolation=cv2.INTER_CUBIC)

    img = hr.astype(np.float32) / 255.0
    # ---- first order ----
    if rng.random() < cfg.blur_prob:
        img = _random_blur(img, rng, cfg.kernel_range, cfg.sigma_range)
    s = _sample_scale(rng, cfg.resize_prob, cfg.resize_range)
    img = _resize(img, rng, w * s, h * s)
    img = _noise(img, rng, cfg.gaussian_noise_prob, cfg.noise_range, cfg.poisson_scale_range, cfg.gray_noise_prob)
    img = _jpeg(img, rng, cfg.jpeg_range)
    # ---- second order (resize is relative to the target LR size) ----
    if rng.random() < cfg.second_blur_prob:
        img = _random_blur(img, rng, cfg.kernel_range, cfg.sigma_range2)
    s = _sample_scale(rng, cfg.resize_prob2, cfg.resize_range2)
    img = _resize(img, rng, tw * s, th * s)
    img = _noise(img, rng, cfg.gaussian_noise_prob2, cfg.noise_range2, cfg.poisson_scale_range2, cfg.gray_noise_prob2)
    # final resize to target size and JPEG, in random order
    if rng.random() < 0.5:
        img = _jpeg(_resize(img, rng, tw, th), rng, cfg.jpeg_range2)
    else:
        img = _resize(_jpeg(img, rng, cfg.jpeg_range2), rng, tw, th)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
