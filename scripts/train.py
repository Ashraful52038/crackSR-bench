"""Train any stage from a config.

    python scripts/train.py --config configs/sr_espcn.yaml
    python scripts/train.py --config configs/seg_unet.yaml --opts train.epochs=5 out_dir=runs/debug
"""
import argparse

import _bootstrap  # noqa: F401

from utils.config import add_config_args, load_config


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    task = cfg["task"]
    if task == "classification":
        from engine import cls as engine
    elif task == "sr":
        from engine import sr as engine
    elif task == "segmentation":
        from engine import seg as engine
    else:
        raise ValueError(f"Unknown task '{task}' in {args.config}")
    engine.train(cfg)


if __name__ == "__main__":
    main()
