"""Evaluate a trained model on its test set (or on other folders).

    # classification: val/test split + every set under data.extra_test (e.g. field images)
    python scripts/evaluate.py --config configs/cls_dinov2.yaml --checkpoint runs/cls_dinov2/best.pt

    # super-resolution on the realistic-degradation test set, saving SR images, with LPIPS
    python scripts/evaluate.py --config configs/sr_swinir.yaml --checkpoint runs/sr_swinir/best.pt \
        --lr_dir data/seg/test/lr_x4_real --save_dir results/sr/swinir --lpips

    # score SR outputs produced by another codebase (e.g. ResShift / SinSR)
    python scripts/evaluate.py --config configs/sr_bicubic.yaml --sr_dir results/sr/resshift --lpips

    # segmentation
    python scripts/evaluate.py --config configs/seg_unet.yaml --checkpoint runs/seg_unet/best.pt
"""
import argparse

import _bootstrap  # noqa: F401

from utils.config import add_config_args, load_config


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--checkpoint", help="best.pt from scripts/train.py (not needed for bicubic or --sr_dir)")
    parser.add_argument("--out", help="Write metrics to this JSON file")
    parser.add_argument("--save_dir", help="SR: save SR images / Seg: save predicted masks")
    parser.add_argument("--lr_dir", help="SR: LR test images (default data.test_lr)")
    parser.add_argument("--hr_dir", help="SR: HR test images (default data.test_hr)")
    parser.add_argument("--sr_dir", help="SR: score pre-computed SR images instead of running a model")
    parser.add_argument("--lpips", action="store_true", help="SR: also compute LPIPS")
    parser.add_argument("--images", help="Seg: test images (default data.test_images)")
    parser.add_argument("--masks", help="Seg: test masks (default data.test_masks)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)

    task = cfg["task"]
    if task == "classification":
        from engine import cls
        cls.evaluate(cfg, args.checkpoint, args.out)
    elif task == "sr":
        from engine import sr
        sr.evaluate(cfg, args.checkpoint, args.lr_dir, args.hr_dir, args.sr_dir, args.save_dir, args.lpips, args.out)
    elif task == "segmentation":
        from engine import seg
        seg.evaluate(cfg, args.checkpoint, args.images, args.masks, args.save_dir, args.out)
    else:
        raise ValueError(f"Unknown task '{task}'")


if __name__ == "__main__":
    main()
