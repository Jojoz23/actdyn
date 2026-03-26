from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def chunked_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = (~mask).float()
    return (values * mask_f).sum() / mask_f.sum().clamp_min(eps)


def batched_l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim) + 1e-12)


def safe_std(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return std.astype(np.float32)


def linear_interpolate(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high <= low:
        return out_high
    alpha = (value - low) / (high - low)
    alpha = min(max(alpha, 0.0), 1.0)
    return out_low + alpha * (out_high - out_low)


def step_from_prefix_condition(scores: np.ndarray, threshold: float, k_min: int, k_max: int) -> int:
    assert scores.ndim == 1
    commit = k_min
    for idx, score in enumerate(scores, start=1):
        if score <= threshold:
            commit = idx
        else:
            break
    return int(max(k_min, min(commit, k_max)))


def maybe_detach_dict(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)
    return out


def clamp_int(value: int, low: int, high: int) -> int:
    return int(max(low, min(value, high)))
