"""Parameter count, MACs and latency, reported in the paper's efficiency table."""
import time

import numpy as np
import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: torch.nn.Module, x: torch.Tensor) -> float | None:
    """Multiply-accumulates for one forward pass (fvcore counts one fused MAC as one 'flop').
    Returns None if fvcore is missing or cannot trace the model."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    try:
        with torch.no_grad():
            fca = FlopCountAnalysis(model.eval(), x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            return float(fca.total())
    except Exception:
        return None


@torch.no_grad()
def measure_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int = 10, runs: int = 50) -> dict:
    model.eval()
    cuda = x.is_cuda
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    out = {"latency_ms": float(np.mean(times)), "latency_std_ms": float(np.std(times))}
    if cuda:
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out
