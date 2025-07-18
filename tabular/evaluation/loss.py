from dataclasses import dataclass
from typing import Self

from torch import Tensor

from tabular.evaluation.inference import Loss


@dataclass
class LossAccumulator:
    loss: float = 0
    n: int = 0

    def update_batch(self, batch_loss: Loss, batch: Tensor):
        examples = len(batch)
        self.loss += batch_loss.loss * examples
        self.n += examples

    @property
    def avg(self) -> float:
        return float(round(self.loss / self.n, 4))

    def __add__(self, other) -> Self:
        return LossAccumulator(loss=self.loss + other.loss, n=self.n + other.n)
