#!/usr/bin/env bash
# Download pretrained SR weights into weights/ (git-ignored).
set -euo pipefail
mkdir -p weights
cd weights

get() {
  if [ -f "$2" ]; then echo "exists: $2"; else echo "downloading: $2"; wget -q --show-progress -O "$2" "$1"; fi
}

# Real-ESRGAN x4 (RRDBNet) - configs/sr_realesrgan.yaml
get https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth RealESRGAN_x4plus.pth

# SwinIR-M x4, real-world degradations, PSNR-oriented - configs/sr_swinir.yaml
get https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth SwinIR-M_x4_realSR_PSNR.pth

# SwinIR-M x4, classical bicubic SR - for the bicubic-degradation experiment
get https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth SwinIR-M_x4_classical.pth

cat <<'EOF'

Manual downloads (hosted on Google Drive / HF, links in each repo's README):
  HAT    https://github.com/XPixelGroup/HAT          -> weights/HAT_SRx4_ImageNet-pretrain.pth  (configs/sr_hat.yaml)
  SPAN   https://github.com/hongyuanyu/SPAN          -> weights/spanx4_ch48.pth                  (configs/sr_span.yaml)
  MambaIR https://github.com/csguoh/MambaIR          -> run its own inference, evaluate with --sr_dir
  ResShift https://github.com/zsyOAOA/ResShift       -> run its own inference, evaluate with --sr_dir
  SinSR  https://github.com/wyf0912/SinSR            -> run its own inference, evaluate with --sr_dir
EOF
