import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

from .utils import rotate_half


class Linear(Module):
    in_features: int
    out_feature: int
    weights: Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_feature = out_features

        self.weights = Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))

        self._reset_params()

    def _reset_params(self):
        std = (2 / (self.in_features + self.out_feature)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights.T


class Embedding(Module):
    num_embedding: int
    embedding_dim: int
    embeds: Parameter

    def __init__(
        self,
        num_embedding: int,
        embdding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.num_embedding = num_embedding
        self.embedding_dim = embdding_dim
        self.embeds = Parameter(torch.empty((num_embedding, embdding_dim), dtype=dtype, device=device))

    def _reset_param(self):
        nn.init.trunc_normal_(self.embeds)

    def forward(self, token_ids: Tensor):
        out_shape = (*token_ids.shape, self.embedding_dim)
        return self.embeds[token_ids.flatten(), :].reshape(out_shape)


class RMSNorm(Module):
    d_model: int
    eps: float
    weights: Parameter

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weights = Parameter(torch.empty((d_model,), dtype=dtype, device=device))

    def _reset_param(self):
        nn.init.constant_(self.weights, 1)

    def forward(self, x: Tensor):
        dtype = x.dtype
        x = x.float()
        rms = ((x * x).sum(-1, keepdim=True) / self.d_model + self.eps) ** 0.5
        return (x / rms).to(dtype) * self.weights


class RotaryPositionalEmbedding(Module):
    theta: int
    in_features: int
    max_deq_len: int
    cos: Tensor
    sin: Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.in_features = d_k
        self.max_deq_len = max_seq_len

        inv_freq = (
            theta ** (torch.arange(0, self.in_features, 2, dtype=torch.float32, device=device) / -self.in_features)
        ).unsqueeze(1)
        inv_freq = inv_freq.expand(-1, 2).reshape(1, self.in_features)
        angles = torch.arange(self.max_deq_len, device=device, dtype=torch.float32).unsqueeze(1)
        angles = angles @ inv_freq

        cos = angles.cos()
        sin = angles.sin()
        # cos, sin: shape of [max_seq_len, in_features]

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: Tensor, position_ids: Tensor):
        cos, sin = self.cos[position_ids, :], self.sin[position_ids, :]
        return x * cos + rotate_half(x) * sin
