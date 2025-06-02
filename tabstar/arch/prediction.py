import torch
import torch.nn as nn

from tabstar.arch.config import D_MODEL


class PredictionHead(nn.Module):

    def __init__(self, input_size: int = D_MODEL):
        super().__init__()
        hidden_size = input_size * 4
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
