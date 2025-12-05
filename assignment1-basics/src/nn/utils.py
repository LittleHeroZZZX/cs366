import math

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def generate_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask = mask.tril(0)
    return mask


def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, t_warmup: int, t_cosine: int) -> float:
    if t < t_warmup:
        lr = lr_max * t / t_warmup
    elif t_warmup <= t <= t_cosine:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - t_warmup) / (t_cosine - t_warmup)))
    else:
        lr = lr_min
    return lr
