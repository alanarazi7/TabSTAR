import numpy as np
import torch
from torch import nn, Tensor

from tabular.tabstar.params.config import TabStarConfig


class NumericalFusion(nn.Module):

    def __init__(self, config: TabStarConfig):
        super().__init__()
        self.config = config
        self.scalar_embedder = nn.Sequential(
            nn.Linear(1, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        self.fusion_block = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=2,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )


    def forward(self, textual_embeddings: Tensor, x_num: np.ndarray) -> Tensor:
        batch_size, seq_len, d_model = textual_embeddings.shape
        x_num = torch.tensor(x_num, dtype=textual_embeddings.dtype, device=textual_embeddings.device)
        num_embeddings = self.scalar_embedder(x_num.unsqueeze(-1))
        assert num_embeddings.shape == textual_embeddings.shape
        fusion_input = torch.stack([textual_embeddings, num_embeddings], dim=2)
        assert fusion_input.shape == (batch_size, seq_len, 2, d_model)
        fusion_input = fusion_input.view(batch_size * seq_len, 2, d_model)
        fused = self.fusion_block(fusion_input)
        fused_embeddings = fused.view(batch_size, seq_len, 2, d_model)
        fused_embeddings = fused_embeddings.mean(dim=2)
        assert fused_embeddings.shape == textual_embeddings.shape
        return fused_embeddings
