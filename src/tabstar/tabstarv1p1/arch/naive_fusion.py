import torch
from torch import nn, Tensor

from tabstar.tabstarv1p1.constants import D_MODEL


class NaiveFusion(nn.Module):

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.ln_text = nn.LayerNorm(d_model)
        self.ln_num = nn.LayerNorm(d_model)
        self.text_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, textual_embeddings: Tensor, num_embeddings: Tensor) -> Tensor:
        text_norm = self.ln_text(textual_embeddings) * self.text_weight.to(dtype=textual_embeddings.dtype)
        num_norm = self.ln_num(num_embeddings)
        fused_embeddings = text_norm + num_norm
        assert fused_embeddings.shape == textual_embeddings.shape == num_embeddings.shape
        return fused_embeddings