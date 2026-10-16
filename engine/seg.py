"""Training and evaluation for Stage 3 (segmentation)."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.schedule import warmup_cosine
from models.losses import DiceBCELoss
from models.segmentation import build_seg_model
from utils.common import get_device, get_logger, load_checkpoint, save_checkpoint, save_json, set_seed, trainable_params
from utils.datasets import SegDataset
from utils.image_io import img2tensor, write_mask
from utils.metrics import SegMeter


def build_model(cfg: dict, checkpoint: str | None = None) -> torch.nn.Module:
    mcfg = dict(cfg["model"])
    if checkpoint:
        mcfg["pretrained"] = False
    model = build_seg_model(mcfg)
    if checkpoint:
        load_checkpoint(checkpoint, model)
    return model


@torch.no_grad()
def predict(model, img: np.ndarray, device, threshold: float = 0.5) -> np.ndarray:
    """RGB uint8 image -> boolean crack mask."""
    model.eval()
    prob = torch.sigmoid(model(img2tensor(img)[None].to(device)))[0, 0]
    return (prob > threshold).cpu().numpy()


@torch.no_grad()
def evaluate_dataset(model, ds, device, threshold=0.5, tolerance=2, save_dir=None) -> dict:
    model.eval()
    meter = SegMeter(tolerance)
    for i in tqdm(range(len(ds)), desc="eval", leave=False):
        item = ds[i]
        prob = torch.sigmoid(model(item["image"][None].to(device)))[0, 0]
        pred = (prob > threshold).cpu().numpy()
        meter.update(pred, item["mask"][0].numpy() > 0.5)
        if save_dir:
            write_mask(Path(save_dir) / f"{item['name']}.png", pred)
    return meter.compute()


def train(cfg: dict) -> dict:
    out_dir = Path(cfg["out_dir"])
    logger = get_logger(out_dir)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device"))
    d, t = cfg["data"], cfg["train"]
    ev = cfg.get("eval", {})
    threshold, tolerance = ev.get("threshold", 0.5), ev.get("tolerance", 2)
    monitor = t.get("monitor", "f1")

    model = build_model(cfg).to(device)
    train_ds = SegDataset(d["train_images"], d["train_masks"], crop_size=d.get("crop_size", 384), augment=True)
    val_ds = SegDataset(d["val_images"], d["val_masks"])
    workers = t.get("num_workers", 4)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    logger.info(f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(trainable_params(model), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    steps = t["epochs"] * len(loader)
    sched = warmup_cosine(opt, steps, t.get("warmup_epochs", 1) * len(loader))
    loss_fn = DiceBCELoss(t.get("bce_weight", 0.5), t.get("pos_weight"))
    use_amp = t.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = -1.0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        if epoch % t.get("val_every", 1) and epoch != t["epochs"]:
            continue
        val_m = evaluate_dataset(model, val_ds, device, threshold, tolerance)
        history.append({"epoch": epoch, "loss": running / len(loader), **val_m})
        logger.info(f"epoch {epoch} | loss {running / len(loader):.4f} | val IoU {val_m['iou']:.4f} F1 {val_m['f1']:.4f}")
        if val_m[monitor] > best:
            best = val_m[monitor]
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch=epoch, val=val_m)
    save_checkpoint(out_dir / "last.pt", model, cfg, epoch=t["epochs"])

    results = {}
    if d.get("test_images"):
        load_checkpoint(out_dir / "best.pt", model)
        results["test"] = evaluate_dataset(model, SegDataset(d["test_images"], d["test_masks"]), device, threshold, tolerance)
        logger.info(f"[best] test IoU {results['test']['iou']:.4f} F1 {results['test']['f1']:.4f}")
    save_json({"history": history, "results": results}, out_dir / "metrics.json")
    return results


def evaluate(cfg: dict, checkpoint: str, images=None, masks=None, save_dir=None, out_json=None) -> dict:
    logger = get_logger()
    device = get_device(cfg.get("device"))
    d, ev = cfg["data"], cfg.get("eval", {})
    model = build_model(cfg, checkpoint).to(device)
    ds = SegDataset(images or d["test_images"], masks or d["test_masks"])
    m = evaluate_dataset(model, ds, device, ev.get("threshold", 0.5), ev.get("tolerance", 2), save_dir)
    logger.info("  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    if out_json:
        save_json(m, out_json)
    return m
