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
