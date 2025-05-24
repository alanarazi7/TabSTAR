import torch
import torch.nn as nn

from tabstar.arch.config import D_MODEL


class InteractionEncoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = D_MODEL, num_heads_factor: int = 64,
                 ffn_d_hidden_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        dim_feedforward = d_model * ffn_d_hidden_multiplier
        num_heads = d_model // num_heads_factor
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
