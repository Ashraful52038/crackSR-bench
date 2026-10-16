"""Create a fixed low-resolution test/val set from HR images.

Every SR model is evaluated on the same LR images, so the comparison is fair.
Each image gets its own seed (seed + index), so the output is reproducible.

    python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_bicubic --scale 4 --mode bicubic
    python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_real --scale 4 \
        --degradation configs/degradation_real.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401

from utils.config import load_yaml
from utils.degradation import DegradationConfig, degrade
from utils.image_io import list_images, mod_crop, read_rgb, write_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--mode", choices=["bicubic", "real"], help="Overrides the mode in --degradation")
    parser.add_argument("--degradation", help="YAML with DegradationConfig fields (default: bicubic)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = load_yaml(args.degradation) if args.degradation else {"mode": "bicubic"}
    if args.mode:
        spec["mode"] = args.mode
    cfg = DegradationConfig.from_any(spec, args.scale)

    files = list_images(args.hr_dir)
    out_dir = Path(args.out_dir)
    for i, path in enumerate(tqdm(files, desc=f"{cfg.mode} x{cfg.scale}")):
        hr = mod_crop(read_rgb(path), cfg.scale)
        lr = degrade(hr, cfg, np.random.default_rng(args.seed + i))
        write_rgb(out_dir / f"{path.stem}.png", lr)  # PNG: no extra compression on top
    print(f"Wrote {len(files)} LR images to {out_dir}")


if __name__ == "__main__":
    main()
