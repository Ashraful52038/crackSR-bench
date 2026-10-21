# Data layout

Datasets are not committed (see `.gitignore`). Keep them on Google Drive and link
or copy them here. All paths in `configs/` assume this layout:

```
data/
├── ccic/                  # Concrete Crack Images for Classification (Özgenel, 2019)
│   ├── Negative/
│   └── Positive/
├── field/                 # real field images, same class folders (out-of-domain test)
│   ├── Negative/
│   └── Positive/
└── seg/                   # crack segmentation dataset (e.g. CRACK500, DeepCrack, CrackForest)
    ├── train/{images,masks}
    ├── val/{images,masks,lr_x4_real}
    └── test/{images,masks,lr_x4_bicubic,lr_x4_real}
```

- Masks are binary images (crack > 127) with the same file stem as the image;
  `001.jpg` may pair with `001.png` or `001_mask.png`.
- Get a segmentation dataset into `seg/` with `scripts/prepare_seg_data.py`. DeepCrack
  (537 images, 544x384, train/test split; 10% of train becomes val):

```bash
wget https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip
unzip -q DeepCrack.zip -d DeepCrack
python scripts/prepare_seg_data.py \
    --train_images DeepCrack/train_img --train_masks DeepCrack/train_lab \
    --test_images DeepCrack/test_img --test_masks DeepCrack/test_lab --out data/seg
```

  CRACK500, GAPs384, CFD, AEL and CrackTree200 come as one zip from
  https://github.com/fyangneil/pavement-crack-detection (Google Drive link in its README;
  `pip install gdown && gdown 13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD`).
- The SR models are trained on `seg/train/images` (HR only; LR is synthesised on the fly),
  so SR and segmentation use the same crack images and the same test split.
- `lr_x4_*` folders are generated, never edited by hand:

```bash
python scripts/degrade.py --hr_dir data/seg/val/images  --out_dir data/seg/val/lr_x4_real   --scale 4 --degradation configs/degradation_real.yaml
python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_real  --scale 4 --degradation configs/degradation_real.yaml
python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_bicubic --scale 4 --mode bicubic
```

On Colab, with the data in `MyDrive/crackSR-data/` (same sub-folders):

```bash
for d in ccic field seg; do ln -s /content/drive/MyDrive/crackSR-data/$d data/$d; done
```
