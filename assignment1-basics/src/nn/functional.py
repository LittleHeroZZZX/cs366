import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1):
    x_temp = x - torch.amax(x, dim=dim, keepdim=True)
    exp = x_temp.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_production_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None):
    batch_size, *_, target_len, head_dim = query.shape
    src_len = value.shape[-2]

    if attn_mask is not None:
        assert attn_mask.dtype is torch.bool, "User provided attn_mask must be bool tensor"
        assert attn_mask.shape[-2:] == (target_len, src_len), (
            f"The last two dim of attn_mask should be {(target_len, src_len)}, but get {attn_mask.shape[-2:]}"
        )

    score = (query @ key.mT) / head_dim**0.5
    if attn_mask is not None:
        score = torch.where(attn_mask, score, torch.full_like(score, -torch.inf))

    return softmax(score) @ value
