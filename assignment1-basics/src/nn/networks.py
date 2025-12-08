from typing import Self

import torch
from torch import Tensor, nn

from .basic import Embedding, Linear, MultiheadSelfAttention, RMSNorm
from .config import Config


class SwiGLU(nn.Module):
    d_model: int
    d_ff: int

    def __init__(
        self,
        in_features: int,
        ff_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = in_features
        self.d_ff = ff_features
        self.linear1 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype,
        )
        self.linear2 = Linear(
            self.d_model,
            self.d_ff,
            device=device,
            dtype=dtype,
        )
        self.linear3 = Linear(
            self.d_ff,
            self.d_model,
            device=device,
            dtype=dtype,
        )

    def _reset_params(self):
        self.linear1._reset_params()
        self.linear2._reset_params()
        self.linear3._reset_params()

    def forward(self, x: Tensor):
        gate = self.linear1(x)
        gate = gate * torch.sigmoid(gate)
        value = self.linear2(x)

        return self.linear3(gate * value)


class TransformerBlock(nn.Module):
    hidden_dim: int
    inner_dim: int  # dim of attn out and ffn input
    max_seq_len: int
    theta: float

    def __init__(
        self,
        hidden_dim: int,
        inner_dim: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm1 = RMSNorm(hidden_dim, **factory_kwargs)
        self.mha = MultiheadSelfAttention(hidden_dim, num_heads, theta, max_seq_len, **factory_kwargs)
        self.norm2 = RMSNorm(hidden_dim, **factory_kwargs)
        self.ffn = SwiGLU(hidden_dim, inner_dim, **factory_kwargs)

    def _reset_params(self):
        self.mha._reset_params()
        self.ffn._reset_params()
        self.norm1._reset_param()
        self.norm2._reset_param()

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        hidden_dim: int,
        inner_dim: int,
        num_heads: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kawrgs = {"device": device, "dtype": dtype}

        self.embed = Embedding(vocab_size, hidden_dim, **factory_kawrgs)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, inner_dim, num_heads, context_length, theta, **factory_kawrgs)
                for _ in range(num_layers)
            ]
        )
        self.post_norm = RMSNorm(hidden_dim, **factory_kawrgs)
        self.lm_head = Linear(hidden_dim, vocab_size, **factory_kawrgs)

    def _reset_params(self):
        self.embed._reset_param()
        for layer in self.layers:
            layer._reset_params()
        self.post_norm._reset_param()
        self.lm_head._reset_params()

    def forward(self, x: Tensor):
        embeds = self.embed(x)
        for layer in self.layers:
            embeds = layer(embeds)
        embeds = self.post_norm(embeds)
        logits = self.lm_head(embeds)
        return logits

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(
            vocab_size=config.model.vocab_size,
            context_length=config.model.context_length,
            num_layers=config.model.num_layers,
            hidden_dim=config.model.hidden_dim,
            inner_dim=config.model.inner_dim,
            num_heads=config.model.num_heads,
            theta=config.model.theta,
            device=torch.device(config.model.device),
            dtype=getattr(torch, config.model.dtype),
        )
