import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int = 0, min_ratio: float = 0.01) -> LambdaLR:
    """Linear warm-up followed by cosine decay to `min_ratio` * base LR. Step once per iteration."""

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)
