from dataclasses import dataclass
from typing import Optional

from torch import Tensor

@dataclass
class Loss:
    loss: float


@dataclass
class InferenceOutput:
    y_pred: Tensor
    loss: Optional[Tensor] = None

    def __post_init__(self):
        if len(self.y_pred.shape) == 2:
            self.y_pred = self.y_pred.squeeze(dim=1)

    @property
    def to_loss(self) -> Loss:
        loss = Loss(loss=self.loss.item())
        return loss