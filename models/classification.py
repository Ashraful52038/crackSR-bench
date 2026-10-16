"""Stage 1: crack / no-crack classification.

- simple_cnn: small CNN baseline of the kind that already reaches ~99% on CCIC.
- Any timm model name, e.g.
    convnextv2_tiny.fcmae_ft_in22k_in1k        (ConvNeXt V2)
    tf_efficientnetv2_s.in21k_ft_in1k          (EfficientNetV2)
    vit_small_patch14_dinov2.lvd142m           (DINOv2; use freeze_backbone for a linear probe)
"""
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        layers, ch = [], 3
        for out in (32, 64, 128, 256):
            layers += [nn.Conv2d(ch, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            ch = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_cls_model(cfg: dict) -> nn.Module:
    name = cfg["name"]
    num_classes = cfg.get("num_classes", 2)
    if name == "simple_cnn":
        return SimpleCNN(num_classes)

    import timm

    model = timm.create_model(name, pretrained=cfg.get("pretrained", True), num_classes=num_classes, **cfg.get("kwargs", {}))
    if cfg.get("freeze_backbone", False):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        n_last = cfg.get("unfreeze_last_blocks", 0)
        if n_last:
            blocks = getattr(model, "blocks", None) or getattr(model, "stages", None)
            if blocks is None:
                raise ValueError(f"unfreeze_last_blocks is not supported for {name}")
            for block in list(blocks)[-n_last:]:
                for p in block.parameters():
                    p.requires_grad = True
            norm = getattr(model, "norm", None)
            if isinstance(norm, nn.Module):
                for p in norm.parameters():
                    p.requires_grad = True
    return model


def resolve_data_config(model: nn.Module, cfg: dict) -> dict:
    """Input size and normalisation the backbone was pretrained with."""
    if isinstance(model, SimpleCNN):
        return {"size": cfg.get("img_size", 224), "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    import timm

    dc = timm.data.resolve_data_config({}, model=model)
    return {"size": cfg.get("img_size") or dc["input_size"][-1], "mean": dc["mean"], "std": dc["std"]}
