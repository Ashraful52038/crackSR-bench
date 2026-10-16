"""Training and evaluation for Stage 1 (classification)."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from engine.schedule import warmup_cosine
from models.classification import build_cls_model, resolve_data_config
from utils.common import get_device, get_logger, load_checkpoint, save_checkpoint, save_json, set_seed, trainable_params
from utils.metrics import classification_metrics


def build_model(cfg: dict, checkpoint: str | None = None) -> nn.Module:
    mcfg = dict(cfg["model"])
    if checkpoint:
        mcfg["pretrained"] = False
    model = build_cls_model(mcfg)
    if checkpoint:
        load_checkpoint(checkpoint, model)
    return model


def build_datasets(cfg: dict, model: nn.Module):
    """Returns train, val, test datasets, extra test sets and class names.

    Either give `data.train_dir/val_dir/test_dir`, or a single `data.root`
    (like CCIC) which is split with a fixed seed so every model sees the same split.
    """
    from torchvision import datasets
    from torchvision import transforms as T

    d = cfg["data"]
    dc = resolve_data_config(model, cfg["model"])
    size, mean, std = dc["size"], dc["mean"], dc["std"]
    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])

    if d.get("train_dir"):
        train = datasets.ImageFolder(d["train_dir"], train_tf)
        val = datasets.ImageFolder(d["val_dir"], eval_tf)
        test = datasets.ImageFolder(d["test_dir"], eval_tf) if d.get("test_dir") else None
        classes = train.classes
    else:
        full_train = datasets.ImageFolder(d["root"], train_tf)
        full_eval = datasets.ImageFolder(d["root"], eval_tf)
        classes = full_train.classes
        perm = np.random.default_rng(d.get("split_seed", 0)).permutation(len(full_train))
        n_test = int(len(perm) * d.get("test_frac", 0.15))
        n_val = int(len(perm) * d.get("val_frac", 0.15))
        test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
        if d.get("train_limit"):  # learning-curve / low-data experiments
            train_idx = train_idx[: d["train_limit"]]
        train = Subset(full_train, train_idx.tolist())
        val = Subset(full_eval, val_idx.tolist())
        test = Subset(full_eval, test_idx.tolist())

    extra = {}
    for name, path in (d.get("extra_test") or {}).items():
        ds = datasets.ImageFolder(path, eval_tf)
        if ds.classes != classes:
            raise ValueError(f"Extra test set '{name}' has classes {ds.classes}, expected {classes}")
        extra[name] = ds
    return train, val, test, extra, classes


@torch.no_grad()
def evaluate_loader(model, loader, device, classes) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_pred.append(logits.argmax(1).cpu())
        y_true.append(y)
    return classification_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), classes)


def _loader(ds, cfg, shuffle):
    t = cfg["train"]
    bs = t["batch_size"] if shuffle else t.get("eval_batch_size", t["batch_size"])
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=t.get("num_workers", 4),
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle)


def _evaluate_all(model, cfg, device, val, test, extra, classes) -> dict:
    results = {"val": evaluate_loader(model, _loader(val, cfg, False), device, classes)}
    if test is not None:
        results["test"] = evaluate_loader(model, _loader(test, cfg, False), device, classes)
    for name, ds in extra.items():
        results[name] = evaluate_loader(model, _loader(ds, cfg, False), device, classes)
    return results


def train(cfg: dict) -> dict:
    out_dir = Path(cfg["out_dir"])
    logger = get_logger(out_dir)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device"))
    t = cfg["train"]

    model = build_model(cfg).to(device)
    train_ds, val, test, extra, classes = build_datasets(cfg, model)
    logger.info(f"classes={classes} train={len(train_ds)} val={len(val)} test={len(test) if test else 0} extra={list(extra)}")
    train_loader = _loader(train_ds, cfg, True)

    opt = torch.optim.AdamW(trainable_params(model), lr=t["lr"], weight_decay=t.get("weight_decay", 0.05))
    steps = t["epochs"] * len(train_loader)
    sched = warmup_cosine(opt, steps, t.get("warmup_epochs", 1) * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.1))
    use_amp = t.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, history = -1.0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val_m = evaluate_loader(model, _loader(val, cfg, False), device, classes)
        history.append({"epoch": epoch, "loss": running / len(train_loader), **{k: val_m[k] for k in ("accuracy", "f1")}})
        logger.info(f"epoch {epoch} | loss {history[-1]['loss']:.4f} | val acc {val_m['accuracy']:.4f} f1 {val_m['f1']:.4f}")
        if val_m["f1"] > best:
            best = val_m["f1"]
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch=epoch, val=val_m)
    save_checkpoint(out_dir / "last.pt", model, cfg, epoch=t["epochs"])

    load_checkpoint(out_dir / "best.pt", model)
    results = _evaluate_all(model, cfg, device, val, test, extra, classes)
    for split, m in results.items():
        logger.info(f"[best] {split}: acc {m['accuracy']:.4f} p {m['precision']:.4f} r {m['recall']:.4f} f1 {m['f1']:.4f}")
    save_json({"history": history, "results": results}, out_dir / "metrics.json")
    return results


def evaluate(cfg: dict, checkpoint: str, out_json: str | None = None) -> dict:
    logger = get_logger()
    device = get_device(cfg.get("device"))
    model = build_model(cfg, checkpoint).to(device)
    _, val, test, extra, classes = build_datasets(cfg, model)
    results = _evaluate_all(model, cfg, device, val, test, extra, classes)
    for split, m in results.items():
        logger.info(f"{split}: acc {m['accuracy']:.4f} p {m['precision']:.4f} r {m['recall']:.4f} f1 {m['f1']:.4f}")
    if out_json:
        save_json(results, out_json)
    return results
