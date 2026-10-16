"""Main experiment: does super-resolution help crack segmentation?

LR image --(SR method)--> SR image --(fixed segmentation model trained on HR)--> mask,
scored against the ground truth. For every SR method the output table holds
  - image quality: PSNR, SSIM (vs. the real HR image)
  - downstream:    IoU, F1, precision, recall, tolerance-F1
  - reliability:   hallucinated / erased crack pixels (vs. the HR prediction)
The `hr` method (segmentation on the real HR image) is the upper bound and
`bicubic` the lower bound.

    python scripts/pipeline_eval.py --config configs/pipeline_x4_real.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401

from engine import seg as seg_engine
from engine import sr as sr_engine
from utils.common import get_device, get_logger, save_json
from utils.config import add_config_args, load_config
from utils.image_io import img2tensor, index_folder, list_images, mod_crop, pair_files, read_mask, read_rgb, tensor2img, write_rgb
from utils.metrics import HallucinationMeter, SegMeter, psnr, ssim
from utils.vis import error_overlay, side_by_side


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    logger = get_logger()
    device = get_device(cfg.get("device"))
    d = cfg["data"]
    scale = cfg["scale"]
    threshold, tolerance = cfg.get("threshold", 0.5), cfg.get("tolerance", 2)
    crop_border = cfg.get("crop_border", scale)
    vis_dir = Path(cfg["vis_dir"]) if cfg.get("vis_dir") else None
    n_vis = cfg.get("n_vis", 8)

    seg_cfg = load_config(cfg["seg"]["config"])
    seg_model = seg_engine.build_model(seg_cfg, cfg["seg"]["checkpoint"]).to(device).eval()

    hr_files = list_images(d["hr_images"])[: cfg.get("limit")]
    mask_files = pair_files(hr_files, d["masks"])
    lr_files = pair_files(hr_files, d["lr_images"])
    logger.info(f"{len(hr_files)} test images, x{scale}")

    # HR images, GT masks and the HR prediction (reference for hallucination) are shared by all methods
    samples = []
    for hp, mp, lp in tqdm(list(zip(hr_files, mask_files, lr_files)), desc="HR reference"):
        hr = mod_crop(read_rgb(hp), scale)
        gt = mod_crop(read_mask(mp), scale).astype(bool)
        if gt.shape != hr.shape[:2]:
            raise ValueError(f"Mask {mp.name} size {gt.shape} != image size {hr.shape[:2]}")
        samples.append({"name": hp.stem, "hr": hr, "gt": gt, "lr_path": lp,
                        "hr_pred": seg_engine.predict(seg_model, hr, device, threshold)})

    rows = []
    for method in cfg["methods"]:
        name, mtype = method["name"], method["type"]
        model, sr_index = None, None
        if mtype == "model":
            sr_cfg = load_config(method["config"])
            if sr_cfg["data"]["scale"] != scale:
                raise ValueError(f"{name}: config scale {sr_cfg['data']['scale']} != pipeline scale {scale}")
            model = sr_engine.build_model(sr_cfg, method.get("checkpoint")).to(device).eval()
        elif mtype == "folder":
            sr_index = index_folder(method["sr_dir"])
        elif mtype != "hr":
            raise ValueError(f"{name}: type must be hr, model or folder")

        seg_m, hall_m = SegMeter(tolerance), HallucinationMeter(tolerance)
        psnrs, ssims = [], []
        for i, s in enumerate(tqdm(samples, desc=name)):
            hr, gt = s["hr"], s["gt"]
            if mtype == "hr":
                img = hr
            elif mtype == "model":
                lr = img2tensor(read_rgb(s["lr_path"]))[None].to(device)
                img = tensor2img(sr_engine.sr_forward(model, lr, scale, method.get("tile", 0))[0])
            else:
                if s["name"] not in sr_index:
                    raise FileNotFoundError(f"{name}: no SR output for {s['name']} in {method['sr_dir']}")
                img = read_rgb(sr_index[s["name"]])
            h, w = min(img.shape[0], hr.shape[0]), min(img.shape[1], hr.shape[1])
            img, hr_c, gt_c, ref = img[:h, :w], hr[:h, :w], gt[:h, :w], s["hr_pred"][:h, :w]

            pred = seg_engine.predict(seg_model, img, device, threshold)
            seg_m.update(pred, gt_c)
            hall_m.update(pred, gt_c, ref)
            if mtype != "hr":
                psnrs.append(psnr(img, hr_c, crop_border))
                ssims.append(ssim(img, hr_c, crop_border))
            if vis_dir and i < n_vis:
                slug = name.lower().replace(" ", "_")
                write_rgb(vis_dir / slug / f"{s['name']}.png", side_by_side([img, error_overlay(img, pred, gt_c)]))

        row = {"method": name,
               "psnr": float(np.mean(psnrs)) if psnrs else np.nan,
               "ssim": float(np.mean(ssims)) if ssims else np.nan,
               **seg_m.compute(), **hall_m.compute()}
        rows.append(row)
        logger.info(f"{name}: PSNR {row['psnr']:.2f} SSIM {row['ssim']:.4f} | IoU {row['iou']:.4f} F1 {row['f1']:.4f} "
                    f"| halluc {row['halluc_rate']:.4f} erased {row['erased_rate']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = Path(cfg.get("out_csv", "results/pipeline.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    save_json({"config": cfg, "rows": rows}, out_csv.with_suffix(".json"))
    cols = ["method", "psnr", "ssim", "iou", "f1", f"tol{tolerance}_f1", "halluc_rate", "erased_rate"]
    print(df[cols].to_markdown(index=False, floatfmt=".4f"))
    logger.info(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
