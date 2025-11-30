import torch
from torch import Tensor, nn

from .basic import Linear, RMSNorm, MultiheadSelfAttention


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
    