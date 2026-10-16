"""YAML config loading with command-line overrides (``--opts train.lr=1e-4``)."""
import argparse
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = _parse_value(raw)
    return cfg


def _parse_value(raw: str):
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML reads "3e-4" (no dot) as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    parser.add_argument(
        "--opts", nargs="*", default=[],
        help="Override config values, e.g. --opts train.lr=1e-4 out_dir=runs/test",
    )
    return parser
