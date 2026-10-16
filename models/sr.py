"""Stage 2: single-image super-resolution. Inputs and outputs are RGB in [0, 1].

Built in:
- bicubic   : interpolation baseline (no parameters).
- espcn     : ESPCN (Shi et al., CVPR 2016), the baseline of the uploaded paper.
- rrdbnet   : RRDBNet, the Real-ESRGAN generator. Parameter names match BasicSR,
              so RealESRGAN_x4plus.pth loads directly through `init_weights`.

Through spandrel (`name: spandrel`, `weights: <file>`):
  SwinIR, HAT, SPAN, OmniSR, DAT, SRFormer, ... - the architecture is detected
  from the checkpoint, and the model can then be fine-tuned like any other.

Models that need their own code (diffusion SR such as ResShift / SinSR,
MambaIR if your spandrel version lacks it) are evaluated from a folder of
their outputs: see `sr_dir` in scripts/evaluate.py and scripts/pipeline_eval.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bicubic(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False).clamp(0, 1)


class ESPCN(nn.Module):
    def __init__(self, scale: int = 4, in_ch: int = 3, feats: int = 64, residual: bool = False):
        super().__init__()
        self.scale, self.residual = scale, residual
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, feats, 5, padding=2), nn.Tanh(),
            nn.Conv2d(feats, feats // 2, 3, padding=1), nn.Tanh(),
            nn.Conv2d(feats // 2, in_ch * scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        out = self.body(x)
        if self.residual:
            out = out + F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return out


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError("RRDBNet supports scale 1, 2 or 4")
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            x = F.pixel_unshuffle(x, 4)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_weights(model: nn.Module, path: str, strict: bool = True) -> None:
    """Load a BasicSR / Real-ESRGAN style checkpoint or one saved by this repo."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params", "model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=strict)


def load_spandrel(path: str) -> tuple[nn.Module, int]:
    from spandrel import ModelLoader

    desc = ModelLoader().load_from_file(path)
    return desc.model, desc.scale


def build_sr_model(cfg: dict, scale: int) -> nn.Module:
    name = cfg["name"].lower()
    kwargs = cfg.get("kwargs", {})
    if name == "bicubic":
        return Bicubic(scale)
    if name == "espcn":
        model = ESPCN(scale=scale, **kwargs)
    elif name == "rrdbnet":
        model = RRDBNet(scale=scale, **kwargs)
    elif name == "spandrel":
        model, model_scale = load_spandrel(cfg["weights"])
        if model_scale != scale:
            raise ValueError(f"{cfg['weights']} is a x{model_scale} model but data.scale is {scale}")
        return model
    else:
        raise ValueError(f"Unknown SR model: {cfg['name']}")
    if cfg.get("init_weights"):
        load_weights(model, cfg["init_weights"])
    return model
