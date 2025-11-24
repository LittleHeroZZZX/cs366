import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch import Tensor


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
