import torch
from pandas import Series

from tabstar.training.metrics import calculate_loss


def test_regression_loss():
    predictions = torch.tensor([[2.5], [0.0], [2.1]])
    y = Series([3.0, -0.5, 2.0])
    loss = calculate_loss(predictions, y, d_output=1)
    assert round(loss.item(), 4) == 0.17

def test_multiclass_loss():
    predictions = torch.tensor([[1.0, 2.0, 0.5, -1.0], [2.0, 0.5, 1.0, -0.5]])
    y = Series([1, 2])
    loss = calculate_loss(predictions, y, d_output=10)
    assert round(loss.item(), 4) == 1.0049

def test_binary_class_loss():
    predictions = torch.tensor([[2.0, 1.0], [0.5, 2.5]])
    y = Series([0, 1])
    loss = calculate_loss(predictions, y, d_output=2)
    assert round(loss.item(), 4) == 0.2201