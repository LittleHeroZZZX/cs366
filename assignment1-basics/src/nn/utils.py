import math
import os
import typing
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
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


@torch.no_grad()
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps=1e-6) -> None:
    grads: tuple = tuple(p.grad for p in params if p.grad is not None)
    square_sum = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        square_sum += torch.sum(g**2)
    norm_all = square_sum.sqrt()
    if norm_all > max_norm:
        clip_coef = max_norm / (norm_all + eps)
        for g in grads:
            g.mul_(clip_coef)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    states = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(states, out)


def load_checkpoint(
    checkpoint: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    states = torch.load(checkpoint)
    model.load_state_dict(states["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(states["optimizer_state_dict"])
    iteration = states["iteration"]
    return iteration


def load_batch(dataset: npt.NDArray, batch_size: int, context_size: int, device: torch.device):
    max_start_index = dataset.shape[0] - context_size + 1
    start_indices = np.random.randint(0, max_start_index - 1, size=batch_size)
    x_batch = np.stack([dataset[i : i + context_size] for i in start_indices])
    y_batch = np.stack([dataset[i + 1 : i + context_size + 1] for i in start_indices])
    x_tensor = torch.tensor(x_batch, device=device, dtype=torch.int)
    y_tensor = torch.tensor(y_batch, device=device, dtype=torch.int)
    return x_tensor, y_tensor
