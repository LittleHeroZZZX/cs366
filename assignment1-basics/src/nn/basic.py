import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter

from . import functional as F
from .utils import generate_causal_mask, rotate_half


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
        return x @ self.weights.mT


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

    def forward(self, x: Tensor, position_ids: Tensor | None = None):
        input_dtype = x.dtype
        if position_ids is None:
            position_ids = torch.arange(x.shape[-2], device=x.device)
        cos, sin = self.cos[position_ids, :], self.sin[position_ids, :]
        return (x * cos + rotate_half(x) * sin).to(input_dtype)


class MultiheadSelfAttention(Module):
    embed_dim: int
    num_heads: int
    head_dim: int
    theta: float | None
    max_seq_len: int | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Expect embed_dim % num_heads == 0, but got {embed_dim % num_heads}"

        assert not ((theta is None) ^ (max_seq_len is None)), (
            "You must provide both theta and max_seq_len if intending to enable RoPE"
        )

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.qkv_proj = Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
        self.out_proj = Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.rotery_embed = None
        if theta is not None and max_seq_len is not None:
            self.rotery_embed = RotaryPositionalEmbedding(
                theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device
            )

    def _reset_params(self):
        self.qkv_proj._reset_params()
        self.out_proj._reset_params()

    def forward(self, x: Tensor):
        seq_len = x.shape[-2]
        qkv: Tensor = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, -1)
        last_two_dim = (self.num_heads, self.head_dim)
        q, k, v = q.unflatten(-1, last_two_dim), k.unflatten(-1, last_two_dim), v.unflatten(-1, last_two_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (bs, num_heads, seq_len, head_dim)
        if self.rotery_embed is not None:
            q, k = self.rotery_embed(q), self.rotery_embed(k)

        causal_mask = generate_causal_mask(seq_len, device=x.device)

        attn_out = F.scaled_dot_production_attention(q, k, v, causal_mask)
        attn_out = attn_out.transpose(1, 2).flatten(-2, -1)
        out = self.out_proj(attn_out)
        return out
