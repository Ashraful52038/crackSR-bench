# crackSR-bench

Super-resolution and segmentation for crack detection in infrastructure images.

The paper has three stages and one central question: **does super-resolution actually
help crack detection, and can it invent cracks that are not there?**

| Stage | Task | Models |
|---|---|---|
| 1 | Crack / no-crack classification (CCIC + field images) | Simple CNN, ConvNeXt V2, EfficientNetV2, DINOv2 + linear head |
| 2 | Super-resolution x4 (main contribution) | Bicubic, ESPCN (baseline paper), Real-ESRGAN, SwinIR, HAT, SPAN, plus external outputs from MambaIR / ResShift / SinSR |
| 3 | Crack segmentation | U-Net, UNet++, SegFormer |
| 2→3 | LR → SR → segmentation | IoU/F1 after SR and hallucination analysis (`scripts/pipeline_eval.py`) |

Design choices that answer typical reviewer questions:

- **Fair comparison.** All SR models use the same crack images, the same degradation and the same fixed LR test set (`scripts/degrade.py`).
- **Realistic degradation.** Besides bicubic downscaling, LR images are made with a Real-ESRGAN-style second-order blur, resize, noise and JPEG pipeline (`configs/degradation_real.yaml`).
- **Downstream proof.** One segmentation model, trained on HR images, is applied to every SR output. This measures whether SR improves IoU/F1, not only PSNR.
- **Hallucination.** Crack pixels predicted on the SR image that are absent from both the ground truth and the HR prediction are counted (`halluc_rate`), as are cracks that SR erased (`erased_rate`).
- **Efficiency.** Params, MACs, latency and peak memory per model (`scripts/benchmark_efficiency.py`).

## Repository layout

```
crackSR-bench/
├── configs/      one YAML per experiment (+ degradation_real.yaml, pipeline_x4_real.yaml)
├── data/         datasets, git-ignored; see data/README.md for the layout
├── engine/       training / evaluation loops: cls.py, sr.py, seg.py
├── models/       classification.py, sr.py, segmentation.py, losses.py
├── utils/        datasets, degradation, metrics, efficiency, image I/O, visualisation
├── scripts/      train.py, evaluate.py, degrade.py, pipeline_eval.py,
│                 benchmark_efficiency.py, make_table.py, download_weights.sh
├── notebooks/    colab_quickstart.ipynb
├── results/      metric tables (CSV/JSON)
└── weights/      pretrained SR weights, git-ignored (scripts/download_weights.sh)
```

## Setup

```bash
git clone https://github.com/<you>/crackSR-bench.git
cd crackSR-bench
pip install -r requirements.txt
bash scripts/download_weights.sh        # Real-ESRGAN + SwinIR weights into weights/
```

Run every command from the repository root. On Colab, open
`notebooks/colab_quickstart.ipynb`. Datasets live on Google Drive and are linked into `data/`.

Any config value can be overridden from the command line:

```bash
python scripts/train.py --config configs/sr_espcn.yaml --opts train.iters=5000 out_dir=runs/espcn_short
```

## Experiments

### 0. Build the fixed LR sets (once)

```bash
python scripts/degrade.py --hr_dir data/seg/val/images  --out_dir data/seg/val/lr_x4_real     --degradation configs/degradation_real.yaml
python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_real    --degradation configs/degradation_real.yaml
python scripts/degrade.py --hr_dir data/seg/test/images --out_dir data/seg/test/lr_x4_bicubic --mode bicubic
```

### 1. Classification

```bash
python scripts/train.py --config configs/cls_simplecnn.yaml
python scripts/train.py --config configs/cls_convnextv2.yaml
python scripts/train.py --config configs/cls_efficientnetv2.yaml
python scripts/train.py --config configs/cls_dinov2.yaml
```

`runs/<name>/metrics.json` holds val, test and `field` (out-of-domain) results. CCIC is
close to saturated (~99%), so the gap on `field` images is the interesting number.

### 2. Super-resolution

```bash
python scripts/train.py --config configs/sr_bicubic.yaml      # validation only
python scripts/train.py --config configs/sr_espcn.yaml        # from scratch
python scripts/train.py --config configs/sr_realesrgan.yaml   # fine-tune RealESRGAN_x4plus
python scripts/train.py --config configs/sr_swinir.yaml       # fine-tune SwinIR-M
python scripts/train.py --config configs/sr_span.yaml         # lightweight (manual weights)
python scripts/train.py --config configs/sr_hat.yaml          # heavy (manual weights)

python scripts/evaluate.py --config configs/sr_swinir.yaml --checkpoint runs/sr_swinir/best.pt \
    --save_dir results/sr/swinir --lpips
```

**Models with their own code (MambaIR, ResShift, SinSR).** Run their official inference on
`data/seg/test/lr_x4_real`, save one PNG per image with the same file name, then:

```bash
python scripts/evaluate.py --config configs/sr_bicubic.yaml --sr_dir results/sr/resshift --lpips
```

### 3. Segmentation

```bash
python scripts/train.py --config configs/seg_unet.yaml
python scripts/train.py --config configs/seg_unetpp.yaml
python scripts/train.py --config configs/seg_segformer.yaml
```

### 2→3. Does SR help segmentation? (main table)

Edit the `methods` list in `configs/pipeline_x4_real.yaml` (checkpoints and external
`sr_dir` folders), then:

```bash
python scripts/pipeline_eval.py --config configs/pipeline_x4_real.yaml
python scripts/make_table.py results/pipeline_x4_real.csv \
    --cols method psnr ssim iou f1 tol2_f1 halluc_rate erased_rate --latex
```

The output has one row per method, with `HR (upper bound)` and `Bicubic` as references.
Colour-coded overlays go to `results/vis/` (green = correct, red = false crack, blue = missed crack).

### Efficiency

```bash
python scripts/benchmark_efficiency.py --configs configs/sr_*.yaml  --size 128 --out results/efficiency_sr.csv
python scripts/benchmark_efficiency.py --configs configs/seg_*.yaml --size 512 --out results/efficiency_seg.csv
```

## Metrics

| | Metric | Notes |
|---|---|---|
| SR | PSNR, SSIM | Y channel, `crop_border = scale`, same protocol as BasicSR |
| SR | LPIPS | `--lpips`, AlexNet |
| Seg | IoU, F1, precision, recall | pixel level, summed over the test set |
| Seg | mIoU / mF1 per image | mean over images |
| Seg | tol2 P/R/F1 | prediction within 2 px of a crack counts as correct |
| SR→Seg | `halluc_rate` | share of predicted crack pixels on the SR image that are > 2 px from both GT and the HR prediction |
| SR→Seg | `erased_rate` | share of cracks found on HR (and in the GT) that disappear after SR |

## Compute notes (free Colab T4)

- Bicubic, ESPCN, SPAN, U-Net and SegFormer-b0/b2 train comfortably.
- Real-ESRGAN and SwinIR: fine-tune from the pretrained weights (batch 4 to 8, `eval.tile` set).
- HAT-L and diffusion SR: do not train on free Colab. Use the pretrained weights, or fine-tune briefly with batch 2.
- Real-ESRGAN is fine-tuned with L1 only (no GAN loss), which reduces hallucination. A GAN-loss variant is a natural ablation.

## Not implemented yet

- SAM 2 fine-tuning for segmentation.
- In-repo MambaIR / diffusion SR training (evaluated from their output folders instead).

## License

MIT (see `LICENSE`).
# crackSR-bench
