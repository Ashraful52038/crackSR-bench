"""Stage 3: crack segmentation (binary, one logit channel).

- unet / unetplusplus : segmentation_models_pytorch with an ImageNet encoder (baselines).
- segformer           : HuggingFace SegFormer (MiT-b0 ... b5 encoders).

`SegModel` takes RGB in [0, 1] of any size: it normalises, pads to a multiple
of 32 and returns logits at the input resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegModel(nn.Module):
    def __init__(self, net: nn.Module, kind: str, pad_to: int = 32):
        super().__init__()
        self.net, self.kind, self.pad_to = net, kind, pad_to
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std
        ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        if self.kind == "segformer":
            logits = self.net(pixel_values=x).logits  # 1/4 resolution
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            logits = self.net(x)
        return logits[..., :h, :w]


def build_seg_model(cfg: dict) -> SegModel:
    name = cfg["name"].lower()
    pretrained = cfg.get("pretrained", True)
    if name in ("unet", "unetplusplus"):
        import segmentation_models_pytorch as smp

        arch = smp.Unet if name == "unet" else smp.UnetPlusPlus
        net = arch(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=cfg.get("encoder_weights", "imagenet") if pretrained else None,
            in_channels=3,
            classes=1,
        )
        return SegModel(net, "smp")
    if name == "segformer":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        hub_id = cfg.get("hub_id", "nvidia/mit-b2")
        if pretrained:
            net = SegformerForSemanticSegmentation.from_pretrained(hub_id, num_labels=1, ignore_mismatched_sizes=True)
        else:  # architecture only, weights come from our checkpoint
            net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(hub_id, num_labels=1))
        return SegModel(net, "segformer")
    raise ValueError(f"Unknown segmentation model: {cfg['name']}")
