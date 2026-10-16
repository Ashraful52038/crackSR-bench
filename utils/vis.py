"""Figures for qualitative comparison."""
import cv2
import numpy as np


def error_overlay(img: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Colour-code a prediction: green = true positive, red = false positive, blue = missed crack."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    color = np.zeros_like(img)
    color[pred & gt] = (0, 255, 0)
    color[pred & ~gt] = (255, 0, 0)
    color[~pred & gt] = (0, 0, 255)
    painted = pred | gt
    out = img.copy()
    out[painted] = (alpha * color[painted] + (1 - alpha) * img[painted]).astype(np.uint8)
    return out


def side_by_side(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    padded = [cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, gap, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in images]
    return np.concatenate(padded, axis=1)[:, :-gap] if gap else np.concatenate(padded, axis=1)
