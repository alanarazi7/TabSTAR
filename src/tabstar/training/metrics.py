from dataclasses import dataclass
from typing import Union, Dict, Optional

import numpy as np
import torch
from numpy.exceptions import AxisError
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from torch import Tensor, softmax
from torch.nn import CrossEntropyLoss, MSELoss


@dataclass
class Metrics:
    score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        self.score = float(self.score)
        self.metrics = {k: float(v) for k, v in self.metrics.items()}


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int, is_pretrain: bool = False) -> Metrics:
    if d_output == 1:
        return _calculate_metrics_for_regression(y_true=y_true, y_pred=y_pred, is_pretrain=is_pretrain)
    elif d_output == 2:
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return _calculate_metrics_for_binary(y_true=y_true, y_pred=y_pred)
    elif d_output > 2:
        return _calculate_metrics_for_multiclass(y_true=y_true, y_pred=y_pred)
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")

def _calculate_metrics_for_regression(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, is_pretrain: bool) -> Metrics:
    rsq = r2_score(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    one_minus_mse = 1 - mse
    metrics = {"r2": rsq, 'mse': mse, '1-mse': one_minus_mse}
    if is_pretrain:
        score = one_minus_mse
    else:
        score = rsq
    return Metrics(score=score, metrics=metrics)

def _calculate_metrics_for_binary(y_true: Union[np.ndarray, Series], y_pred: np.ndarray) -> Metrics:
    auc = roc_auc_score(y_true, y_pred)
    return Metrics(score=auc, metrics={"roc_auc": auc})

def _calculate_metrics_for_multiclass(y_true: Union[np.ndarray, Series], y_pred: np.ndarray) -> Metrics:
    try:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
    except (ValueError, AxisError):
        # Error calculating AUC, likely due to class imbalance or insufficient samples
        auc = _per_class_auc(y_true=y_true, y_pred=y_pred)
    return Metrics(score=auc, metrics={"roc_auc_ovr": auc})

def _per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = {}
    for cls in present_classes:
        # Binary ground truth: 1 for the current class, 0 for others
        y_true_binary = (y_true == cls).astype(int)
        # Predicted probabilities for the current class
        y_pred_scores = y_pred[:, int(cls)]
        try:
            auc = roc_auc_score(y_true_binary, y_pred_scores)
            aucs[cls] = auc
        except ValueError:
            pass
    macro_avg = float(np.mean(list(aucs.values())))
    return macro_avg


def apply_loss_fn(prediction: Tensor, d_output: int) -> Tensor:
    if d_output == 1:
        return prediction.to(torch.float32)
    prediction = prediction.to(torch.float32)
    prediction = softmax(prediction, dim=1)
    if d_output == 2:
        # We want the probability of '1'
        prediction = prediction[:, 1]
    return prediction


def calculate_loss(predictions: Tensor, y: Union[Series, np.ndarray], d_output: int) -> Tensor:
    is_reg = bool(d_output == 1)
    if is_reg:
        loss_fn = MSELoss()
        dtype = torch.float32
    else:
        loss_fn = CrossEntropyLoss()
        dtype = torch.long
    if not isinstance(y, Tensor):
        y = torch.tensor(y, dtype=dtype)
    y = y.to(predictions.device)
    if is_reg and y.ndim == 1:
        y = y.unsqueeze(1)
    loss = loss_fn(predictions, y)
    return loss