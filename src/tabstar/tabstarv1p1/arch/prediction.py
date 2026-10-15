import torch
import torch.nn as nn

from tabstar.tabstarv1p1.constants import D_MODEL, REG_Z_CLIP, REG_DIST_BINS


class PredictionHead(nn.Module):

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        hidden_size = d_model * 4
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualRegressionHead(nn.Module):
    """A scalar sub-head and a distributional sub-head trained jointly; forward concatenates
    their outputs as (..., 1 + n_bins), scalar first, so a downstream consumer can blend both
    predictions at inference instead of relying on either alone."""

    def __init__(self, d_model: int = D_MODEL, reg_dist_bins: int = REG_DIST_BINS):
        super().__init__()
        self.scalar_head = PredictionHead(d_model=d_model)
        self.dist_head = DistributionalRegressionHead(d_model=d_model, reg_dist_bins=reg_dist_bins)

    @property
    def bin_edges(self) -> torch.Tensor:
        return self.dist_head.bin_edges

    @property
    def bin_centers(self) -> torch.Tensor:
        return self.dist_head.bin_centers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalar_out = self.scalar_head(x)
        dist_out = self.dist_head(x)
        return torch.cat([scalar_out, dist_out], dim=-1)


class DistributionalRegressionHead(nn.Module):
    """Predicts a distribution over fixed z-score bins instead of a single scalar."""

    def __init__(self, d_model: int = D_MODEL, reg_dist_bins: int = REG_DIST_BINS):
        super().__init__()
        hidden_size = d_model * 4
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, reg_dist_bins)
        )
        bin_edges = torch.linspace(-REG_Z_CLIP, REG_Z_CLIP, reg_dist_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer("bin_edges", bin_edges, persistent=False)
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
