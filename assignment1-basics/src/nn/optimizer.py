import math
from collections.abc import Callable
from typing import cast

import torch
from torch.optim.optimizer import ParamsT


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):  # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        self.current_step = 0
        defaults = {"alpha": lr, "beta1": betas[0], "beta2": betas[1], "lambda": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):  # type: ignore
        loss = None if closure is None else closure()
        self.current_step += 1
        t = self.current_step
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lambda_ = group["lambda"]
            eps = group["eps"]
            for p in group["params"]:
                p = cast(torch.nn.Parameter, p)
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p - alpha_t * m / (v**0.5 + eps)
                p.data = p - alpha * lambda_ * p
                state["m"] = m
                state["v"] = v

        return loss
