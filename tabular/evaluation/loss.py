from dataclasses import dataclass
from typing import Callable, Self

import torch
from torch import Tensor, softmax
from torch.nn import MSELoss, CrossEntropyLoss

from tabular.evaluation.inference import Loss
from tabular.preprocessing.objects import SupervisedTask


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



def get_loss_fn(task_type: SupervisedTask) -> Callable:
    if task_type in {SupervisedTask.BINARY, SupervisedTask.MULTICLASS}:
        return CrossEntropyLoss()
    elif task_type == SupervisedTask.REGRESSION:
        return MSELoss()
    raise TypeError(f"Unknown task_type: {task_type}")


def get_torch_dtype(task_type: SupervisedTask) -> torch.dtype:
    if task_type == SupervisedTask.REGRESSION:
        return torch.float32
    else:
        return torch.long


def apply_loss_fn(prediction: Tensor, task_type: SupervisedTask):
    if task_type == SupervisedTask.REGRESSION:
        return prediction
    assert task_type in {SupervisedTask.BINARY, SupervisedTask.MULTICLASS}
    prediction = prediction.to(torch.float32)
    prediction = softmax(prediction, dim=1)
    if task_type == SupervisedTask.BINARY:
        # We want the probability of '1'
        prediction = prediction[:, 1]
    return prediction