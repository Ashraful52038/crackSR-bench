"""Metrics for the three stages.

SR:   PSNR / SSIM on the Y channel with `crop_border` pixels removed (the usual
      SR-paper protocol, matching BasicSR), optional LPIPS.
Seg:  IoU, F1, precision, recall (pixel level, accumulated over the dataset),
      per-image mean IoU/F1, and tolerance F1 (a prediction within `tolerance`
      pixels of a crack counts as correct, common in crack-detection papers).
SR->Seg: hallucination statistics, i.e. cracks that appear on the SR image but
      neither in the ground truth nor on the real HR image.
"""
import cv2
import numpy as np
import torch

# ----------------------------------------------------------------------------- SR


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """RGB in [0, 255] -> Y (ITU-R BT.601) in [16, 235], as MATLAB's rgb2ycbcr."""
    img = img.astype(np.float64)
    return 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0


def _prepare(img: np.ndarray, crop_border: int, y_channel: bool) -> np.ndarray:
    img = img.astype(np.float64)
    if crop_border:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    if y_channel:
        img = rgb_to_y(img)[..., None]
    return img


def psnr(sr: np.ndarray, hr: np.ndarray, crop_border: int = 0, y_channel: bool = True) -> float:
    a, b = _prepare(sr, crop_border, y_channel), _prepare(hr, crop_border, y_channel)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def _ssim_channel(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1 = cv2.filter2D(a**2, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b**2, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean())


def ssim(sr: np.ndarray, hr: np.ndarray, crop_border: int = 0, y_channel: bool = True) -> float:
    a, b = _prepare(sr, crop_border, y_channel), _prepare(hr, crop_border, y_channel)
    return float(np.mean([_ssim_channel(a[..., c], b[..., c]) for c in range(a.shape[2])]))


class LPIPSMetric:
    """Perceptual distance (lower is better). Needs `pip install lpips`."""

    def __init__(self, device, net: str = "alex"):
        import lpips

        self.device = device
        self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, sr: np.ndarray, hr: np.ndarray) -> float:
        def to_t(x):
            t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1.0
            return t.to(self.device)

        return float(self.fn(to_t(sr), to_t(hr)).item())


# ---------------------------------------------------------------------- Segmentation


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


class SegMeter:
    """Accumulates binary segmentation statistics over a dataset."""

    def __init__(self, tolerance: int = 2):
        self.tolerance = tolerance
        self.tp = self.fp = self.fn = 0
        self.img_iou, self.img_f1 = [], []
        self.tol_hit_pred = self.n_pred = self.tol_hit_gt = self.n_gt = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = pred.astype(bool), gt.astype(bool)
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        denom = tp + fp + fn
        # an image with no crack and no prediction is a perfect result
        self.img_iou.append(tp / denom if denom else 1.0)
        self.img_f1.append(2 * tp / (2 * tp + fp + fn) if denom else 1.0)

        self.tol_hit_pred += int(np.logical_and(pred, _dilate(gt, self.tolerance)).sum())
        self.tol_hit_gt += int(np.logical_and(gt, _dilate(pred, self.tolerance)).sum())
        self.n_pred += int(pred.sum())
        self.n_gt += int(gt.sum())

    def compute(self) -> dict:
        eps = 1e-12
        p = self.tp / (self.tp + self.fp + eps)
        r = self.tp / (self.tp + self.fn + eps)
        tp_ = self.tol_hit_pred / (self.n_pred + eps)
        tr_ = self.tol_hit_gt / (self.n_gt + eps)
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "f1": 2 * p * r / (p + r + eps),
            "precision": p,
            "recall": r,
            "miou_img": float(np.mean(self.img_iou)) if self.img_iou else 0.0,
            "mf1_img": float(np.mean(self.img_f1)) if self.img_f1 else 0.0,
            f"tol{self.tolerance}_precision": tp_,
            f"tol{self.tolerance}_recall": tr_,
            f"tol{self.tolerance}_f1": 2 * tp_ * tr_ / (tp_ + tr_ + eps),
        }


class HallucinationMeter:
    """Effect of SR on crack evidence, measured with the same segmentation model.

    hallucinated: crack pixels predicted on the SR image farther than `tolerance`
                  from both the ground truth and the prediction on the real HR image
                  (cracks "invented" by the SR model).
    erased:       crack pixels found on the HR image (and true in the GT) that the
                  SR image no longer shows (cracks removed by SR).
    """

    def __init__(self, tolerance: int = 2, min_area: int = 20):
        self.tolerance, self.min_area = tolerance, min_area
        self.hall_px = self.pred_px = self.erased_px = self.ref_px = 0
        self.n_images = self.images_with_hall = 0

    def update(self, pred_sr: np.ndarray, gt: np.ndarray, pred_hr: np.ndarray) -> None:
        pred_sr, gt, pred_hr = pred_sr.astype(bool), gt.astype(bool), pred_hr.astype(bool)
        allowed = _dilate(gt | pred_hr, self.tolerance)
        hall = int(np.logical_and(pred_sr, ~allowed).sum())
        ref = gt & pred_hr
        erased = int(np.logical_and(ref, ~_dilate(pred_sr, self.tolerance)).sum())
        self.hall_px += hall
        self.pred_px += int(pred_sr.sum())
        self.erased_px += erased
        self.ref_px += int(ref.sum())
        self.n_images += 1
        self.images_with_hall += int(hall >= self.min_area)

    def compute(self) -> dict:
        eps = 1e-12
        n = max(self.n_images, 1)
        return {
            "halluc_rate": self.hall_px / (self.pred_px + eps),
            "halluc_px_per_img": self.hall_px / n,
            "halluc_img_frac": self.images_with_hall / n,
            "erased_rate": self.erased_px / (self.ref_px + eps),
        }


# --------------------------------------------------------------------- Classification


def classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    average = "binary" if len(class_names) == 2 else "macro"
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "classes": class_names,
    }
