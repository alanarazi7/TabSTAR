from torch import nn, Tensor

class NumericalEncoder(nn.Module):

    def __init__(self, d_model: int, d_num: int):
        super().__init__()
        self.scalar_embedder = nn.Sequential(
            nn.Linear(d_num, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x_num: Tensor) -> Tensor:
        return self.scalar_embedder(x_num)
