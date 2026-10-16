"""Training and evaluation for Stage 2 (super-resolution)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.schedule import warmup_cosine
from models.losses import build_sr_loss
from models.sr import build_sr_model
from utils.common import get_device, get_logger, load_checkpoint, save_checkpoint, save_json, set_seed, trainable_params
from utils.datasets import SRPairedDataset, SRTrainDataset
from utils.image_io import index_folder, list_images, mod_crop, read_rgb, tensor2img, write_rgb
from utils.metrics import LPIPSMetric, psnr, ssim


def build_model(cfg: dict, checkpoint: str | None = None) -> torch.nn.Module:
    model = build_sr_model(cfg["model"], cfg["data"]["scale"])
    if checkpoint:
        load_checkpoint(checkpoint, model)
    return model


@torch.no_grad()
def sr_forward(model, x: torch.Tensor, scale: int, tile: int = 0, overlap: int = 16) -> torch.Tensor:
    """Super-resolve `x` (B,3,h,w). With `tile` > 0 the image is processed in
    overlapping tiles to keep memory low (needed for SwinIR/HAT on large images).
    Each tile only contributes its interior (an `overlap // 2` margin is dropped
    except at the image border), so tile-edge artifacts do not reach the output."""
    h, w = x.shape[-2:]
    if tile <= 0 or (h <= tile and w <= tile):
        return model(x).clamp(0, 1)
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    stride, margin = tile - overlap, overlap // 2

    def starts(size):
        last = max(size - tile, 0)
        return list(range(0, last, stride)) + [last]

    out = x.new_zeros(x.shape[0], x.shape[1], h * scale, w * scale)
    for y0 in starts(h):
        for x0 in starts(w):
            patch = x[..., y0:y0 + tile, x0:x0 + tile]
            sr = model(patch)
            ph, pw = patch.shape[-2:]
            # interior of this tile in LR coordinates
            top = y0 + (margin if y0 > 0 else 0)
            left = x0 + (margin if x0 > 0 else 0)
            bottom = y0 + ph - (margin if y0 + ph < h else 0)
            right = x0 + pw - (margin if x0 + pw < w else 0)
            out[..., top * scale:bottom * scale, left * scale:right * scale] = sr[
                ..., (top - y0) * scale:(bottom - y0) * scale, (left - x0) * scale:(right - x0) * scale]
    return out.clamp(0, 1)


def _match(sr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = min(sr.shape[0], hr.shape[0]), min(sr.shape[1], hr.shape[1])
    return sr[:h, :w], hr[:h, :w]


def _summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k != "name"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate_pairs(model, dataset, device, scale, crop_border, tile=0, save_dir=None, lpips_fn=None):
    model.eval()
    rows = []
    for i in tqdm(range(len(dataset)), desc="eval", leave=False):
        item = dataset[i]
        sr = sr_forward(model, item["lr"][None].to(device), scale, tile)
        sr_img, hr_img = _match(tensor2img(sr[0]), tensor2img(item["hr"]))
        row = {"name": item["name"], "psnr": psnr(sr_img, hr_img, crop_border), "ssim": ssim(sr_img, hr_img, crop_border)}
        if lpips_fn is not None:
            row["lpips"] = lpips_fn(sr_img, hr_img)
        if save_dir:
            write_rgb(Path(save_dir) / f"{item['name']}.png", sr_img)
        rows.append(row)
    return _summarise(rows), rows


def evaluate_folder(sr_dir, hr_dir, scale, crop_border, lpips_fn=None):
    """Score SR images produced elsewhere (e.g. ResShift / SinSR official code)."""
    sr_index = index_folder(sr_dir)
    rows = []
    for hr_path in list_images(hr_dir):
        if hr_path.stem not in sr_index:
            raise FileNotFoundError(f"No SR output for {hr_path.name} in {sr_dir}")
        sr_img, hr_img = _match(read_rgb(sr_index[hr_path.stem]), mod_crop(read_rgb(hr_path), scale))
        row = {"name": hr_path.stem, "psnr": psnr(sr_img, hr_img, crop_border), "ssim": ssim(sr_img, hr_img, crop_border)}
        if lpips_fn is not None:
            row["lpips"] = lpips_fn(sr_img, hr_img)
        rows.append(row)
    return _summarise(rows), rows


def _cycle(loader):
    while True:
        yield from loader


def train(cfg: dict) -> dict:
    out_dir = Path(cfg["out_dir"])
    logger = get_logger(out_dir)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device"))
    d, t = cfg["data"], cfg["train"]
    scale = d["scale"]
    crop_border = d.get("crop_border", scale)
    tile = cfg.get("eval", {}).get("tile", 0)

    model = build_model(cfg).to(device)
    val_ds = SRPairedDataset(d["val_lr"], d["val_hr"], scale, limit=d.get("val_limit"))
    params = trainable_params(model)
    if not params:
        logger.info("Model has no trainable parameters (e.g. bicubic); running validation only.")
        summary, _ = evaluate_pairs(model, val_ds, device, scale, crop_border, tile)
        logger.info(f"val PSNR {summary['psnr']:.3f} SSIM {summary['ssim']:.4f}")
        save_checkpoint(out_dir / "best.pt", model, cfg, val=summary)
        save_json({"best_val": summary}, out_dir / "train_summary.json")
        return summary

    train_ds = SRTrainDataset(d["train_hr"], scale, d.get("patch_size", 192), d.get("degradation"), d.get("repeat", 1))
    workers = t.get("num_workers", 4)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    logger.info(f"train images={len(train_ds.files)} val images={len(val_ds)} params={sum(p.numel() for p in params):,}")

    opt = torch.optim.Adam(params, lr=t["lr"], betas=(0.9, 0.99), weight_decay=t.get("weight_decay", 0.0))
    total = t["iters"]
    sched = warmup_cosine(opt, total, t.get("warmup_iters", 0))
    loss_fn = build_sr_loss(t.get("loss", "l1"))
    use_amp = t.get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_every, log_every = t.get("val_every", 1000), t.get("log_every", 100)

    best, history, running = -float("inf"), [], 0.0
    data_iter = _cycle(loader)
    model.train()
    pbar = tqdm(range(1, total + 1), desc="train")
    for it in pbar:
        batch = next(data_iter)
        lr_img = batch["lr"].to(device, non_blocking=True)
        hr_img = batch["hr"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            sr = model(lr_img)
        loss = loss_fn(sr.float(), hr_img)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if t.get("grad_clip"):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()

        if it % log_every == 0:
            pbar.set_postfix(loss=f"{running / log_every:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            running = 0.0
        if it % val_every == 0 or it == total:
            summary, _ = evaluate_pairs(model, val_ds, device, scale, crop_border, tile)
            model.train()
            history.append({"iter": it, **summary})
            logger.info(f"iter {it} | val PSNR {summary['psnr']:.3f} SSIM {summary['ssim']:.4f}")
            if summary["psnr"] > best:
                best = summary["psnr"]
                save_checkpoint(out_dir / "best.pt", model, cfg, iter=it, val=summary)
            save_checkpoint(out_dir / "last.pt", model, cfg, iter=it)
    save_json({"best_val_psnr": best, "history": history}, out_dir / "train_summary.json")
    return {"best_val_psnr": best}


def evaluate(cfg: dict, checkpoint: str | None, lr_dir=None, hr_dir=None, sr_dir=None,
             save_dir=None, use_lpips=False, out_json=None) -> dict:
    """Test-set PSNR/SSIM(/LPIPS). Uses `data.test_lr/test_hr` unless dirs are given.
    With `sr_dir`, scores pre-computed SR images instead of running a model."""
    logger = get_logger()
    device = get_device(cfg.get("device"))
    d = cfg["data"]
    scale = d["scale"]
    crop_border = d.get("crop_border", scale)
    hr_dir = hr_dir or d["test_hr"]
    lpips_fn = LPIPSMetric(device) if use_lpips else None

    if sr_dir:
        summary, rows = evaluate_folder(sr_dir, hr_dir, scale, crop_border, lpips_fn)
    else:
        model = build_model(cfg, checkpoint).to(device)
        ds = SRPairedDataset(lr_dir or d["test_lr"], hr_dir, scale)
        summary, rows = evaluate_pairs(model, ds, device, scale, crop_border, cfg.get("eval", {}).get("tile", 0), save_dir, lpips_fn)
    logger.info("  ".join(f"{k} {v:.4f}" for k, v in summary.items()))
    if out_json:
        save_json({"summary": summary, "per_image": rows}, out_json)
    return summary
