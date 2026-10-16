"""Params, MACs, latency and peak memory for each model (efficiency table).

    python scripts/benchmark_efficiency.py --configs configs/sr_*.yaml --size 128 --out results/efficiency_sr.csv
    python scripts/benchmark_efficiency.py --configs configs/seg_*.yaml --size 512 --out results/efficiency_seg.csv

`--size` is the input side length: the LR size for SR models, the image size otherwise.
Weights are not needed (architectures are built without pretrained weights),
except for spandrel configs, which read the architecture from their weights file.
"""
import argparse
import copy
from pathlib import Path

import pandas as pd
import torch

import _bootstrap  # noqa: F401

from utils.common import get_device
from utils.config import load_config
from utils.efficiency import count_macs, count_params, measure_latency


def build(cfg: dict) -> torch.nn.Module:
    cfg = copy.deepcopy(cfg)
    cfg["model"]["pretrained"] = False
    cfg["model"].pop("init_weights", None)
    task = cfg["task"]
    if task == "sr":
        from engine.sr import build_model
    elif task == "segmentation":
        from engine.seg import build_model
    elif task == "classification":
        from engine.cls import build_model
    else:
        raise ValueError(task)
    return build_model(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--device")
    parser.add_argument("--out", default="results/efficiency.csv")
    args = parser.parse_args()
    device = get_device(args.device)

    rows = []
    for path in args.configs:
        cfg = load_config(path)
        name = Path(path).stem
        try:
            model = build(cfg).to(device).eval()
        except Exception as e:  # e.g. missing spandrel weights file
            print(f"skip {name}: {e}")
            continue
        x = torch.rand(args.batch, 3, args.size, args.size, device=device)
        macs = count_macs(model, x)
        row = {"model": name, "params_M": count_params(model) / 1e6,
               "GMACs": macs / 1e9 if macs is not None else None,
               **measure_latency(model, x, runs=args.runs)}
        rows.append(row)
        print(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\ninput {args.batch}x3x{args.size}x{args.size} on {device}")
    print(df.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
