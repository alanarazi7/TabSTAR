from dataclasses import dataclass
from typing import Self


@dataclass
class LossAccumulator:
    loss: float = 0
    n: int = 0

    def update_batch(self, loss: float, n: int):
        self.loss += loss * n
        self.n += n

    @property
    def avg(self) -> float:
        return float(round(self.loss / self.n, 4))

    def __add__(self, other) -> Self:
        return LossAccumulator(loss=self.loss + other.loss, n=self.n + other.n)