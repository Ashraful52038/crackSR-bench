"""Copy a crack segmentation dataset into the layout used by the configs:

    data/seg/{train,val,test}/{images,masks}

Images and masks are paired by file name; masks are binarised and saved as PNG.
If the dataset has no validation split, `--val_frac` of the training images
is held out with a fixed seed.

    # DeepCrack (https://github.com/yhlleo/DeepCrack, dataset/DeepCrack.zip)
    python scripts/prepare_seg_data.py \
        --train_images DeepCrack/train_img --train_masks DeepCrack/train_lab \
        --test_images DeepCrack/test_img --test_masks DeepCrack/test_lab --out data/seg

    # Datasets that keep images and masks in one folder (e.g. 001.jpg + 001.png)
    python scripts/prepare_seg_data.py --train_images X/traincrop --train_masks X/traincrop \
        --val_images X/valcrop --val_masks X/valcrop --test_images X/testcrop --test_masks X/testcrop \
        --img_ext .jpg --mask_ext .png --out data/seg
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from utils.image_io import IMG_EXTS, file_key, read_mask, read_rgb, write_mask


def collect(folder: str, exts: set[str]) -> dict[str, Path]:
    files = sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No files with extensions {sorted(exts)} in {folder}")
    index = {}
    for p in files:
        key = file_key(p)
        if key in index:
            raise ValueError(f"Two files share the name '{key}' in {folder}: {index[key].name}, {p.name}. "
                             "Use --img_ext / --mask_ext to tell images and masks apart.")
        index[key] = p
    return index


def pairs(images: str, masks: str, img_exts: set[str], mask_exts: set[str]) -> list[tuple[Path, Path]]:
    img_index, mask_index = collect(images, img_exts), collect(masks, mask_exts)
    missing = sorted(set(img_index) - set(mask_index))
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) in {images} have no mask in {masks}, e.g. {missing[:5]}")
    return [(img_index[k], mask_index[k]) for k in sorted(img_index)]


def write_split(items: list[tuple[Path, Path]], out: Path, split: str) -> None:
    img_dir, mask_dir = out / split / "images", out / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    for img_path, mask_path in items:
        img, mask = read_rgb(img_path), read_mask(mask_path)
        if mask.shape != img.shape[:2]:
            raise ValueError(f"Size mismatch: {img_path.name} {img.shape[:2]} vs mask {mask.shape}")
        shutil.copy2(img_path, img_dir / f"{file_key(img_path)}{img_path.suffix.lower()}")
        write_mask(mask_dir / f"{file_key(img_path)}.png", mask)
    print(f"{split:5s}: {len(items)} pairs -> {out / split}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_masks", required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_masks", required=True)
    parser.add_argument("--val_images")
    parser.add_argument("--val_masks")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Used only without --val_images")
    parser.add_argument("--img_ext", nargs="*", help="Image extensions, e.g. .jpg (default: all image types)")
    parser.add_argument("--mask_ext", nargs="*", help="Mask extensions, e.g. .png (default: all image types)")
    parser.add_argument("--out", default="data/seg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img_exts = {e.lower() for e in args.img_ext} if args.img_ext else IMG_EXTS
    mask_exts = {e.lower() for e in args.mask_ext} if args.mask_ext else IMG_EXTS
    out = Path(args.out)
    if any((out / s).exists() for s in ("train", "val", "test")):
        raise FileExistsError(f"{out} already has train/val/test folders; remove them or pick another --out")

    train = pairs(args.train_images, args.train_masks, img_exts, mask_exts)
    test = pairs(args.test_images, args.test_masks, img_exts, mask_exts)
    if args.val_images:
        val = pairs(args.val_images, args.val_masks, img_exts, mask_exts)
    else:
        perm = np.random.default_rng(args.seed).permutation(len(train))
        n_val = max(1, int(round(len(train) * args.val_frac)))
        val = [train[i] for i in sorted(perm[:n_val])]
        train = [train[i] for i in sorted(perm[n_val:])]

    for split, items in (("train", train), ("val", val), ("test", test)):
        write_split(items, out, split)


if __name__ == "__main__":
    main()
