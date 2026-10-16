import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE + soft Dice. Dice counters the strong background/crack imbalance."""

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        pw = None if self.pos_weight is None else torch.tensor(self.pos_weight, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant often used for SR."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


def build_sr_loss(name: str) -> nn.Module:
    if name == "l1":
        return nn.L1Loss()
    if name == "charbonnier":
        return CharbonnierLoss()
    raise ValueError(f"Unknown SR loss: {name}")
