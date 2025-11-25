import torch

from torch import Tensor, nn
from .basic import Linear


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
