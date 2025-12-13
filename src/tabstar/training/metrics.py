from dataclasses import dataclass
from typing import Union, Dict, Optional

import numpy as np
import torch
from numpy.exceptions import AxisError
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, accuracy_score, f1_score, log_loss, recall_score
from torch import Tensor, softmax
from torch.nn import CrossEntropyLoss, MSELoss


LOGLOSS = "logloss"
RMSE = "rmse"
R2 = "r2"
ROC_AUC = "roc_auc"
ROC_AUC_OVR = "roc_auc_ovr"

METRICS_TO_MINIMIZE = [LOGLOSS, RMSE]

@dataclass
class Metrics:
    score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        self.score = float(self.score)
        self.metrics = {k: float(v) for k, v in self.metrics.items()}


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int, is_pretrain: bool = False,
                     metric_name: Optional[None] = None) -> Metrics:
    if d_output == 1:
        return _calculate_metrics_for_regression(y_true=y_true, y_pred=y_pred, is_pretrain=is_pretrain,
                                                 metric_name=metric_name)
    elif d_output == 2:
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return _calculate_metrics_for_binary(y_true=y_true, y_pred=y_pred, metric_name=metric_name)
    elif d_output > 2:
        return _calculate_metrics_for_multiclass(y_true=y_true, y_pred=y_pred, metric_name=metric_name)
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")

def _calculate_metrics_for_regression(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, is_pretrain: bool,
                                      metric_name: Optional[str]) -> Metrics:
    rsq = r2_score(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(mse)
    one_minus_mse = 1 - mse
    metrics = {R2: rsq, 'mse': mse, '1-mse': one_minus_mse, RMSE: rmse}
    if is_pretrain:
        score = one_minus_mse
    elif metric_name == RMSE:
        score = rmse
    else:
        if (metric_name is not None) and metric_name not in [RMSE, R2]:
            print(f"⚠️ Unsupported metric_name for regression! {metric_name}. We want: {RMSE} or {R2}.")
        score = rsq
    return Metrics(score=score, metrics=metrics)

def _calculate_metrics_for_binary(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, metric_name: Optional[str]) -> Metrics:
    y_pred_label = (y_pred > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    logloss = log_loss(y_true, y_pred)
    recall = recall_score(y_true, y_pred_label)
    metrics = {ROC_AUC: auc, LOGLOSS: logloss, 'accuracy': acc, 'f1': f1, 'recall': recall}
    if metric_name is not None and metric_name != ROC_AUC:
        print(f"⚠️ Unsupported metric name: {metric_name}. We want: {ROC_AUC}")
    return Metrics(score=auc, metrics=metrics)

def _calculate_metrics_for_multiclass(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, metric_name: Optional[str]) -> Metrics:
    try:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
    except (ValueError, AxisError):
        # Error calculating AUC, likely due to class imbalance or insufficient samples
        auc = _per_class_auc(y_true=y_true, y_pred=y_pred)
    try:
        logloss = log_loss(y_true, y_pred)
    except Exception:
        logloss = np.nan
    y_pred_label = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true, y_pred_label, average='macro')
    acc = accuracy_score(y_true, y_pred_label)
    metrics = {
        ROC_AUC_OVR: auc,
        LOGLOSS: logloss,
        'macro_f1': f1,
        'accuracy': acc,
    }
    if metric_name == LOGLOSS:
        score = logloss
    else:
        if metric_name is not None and metric_name not in [ROC_AUC_OVR, LOGLOSS]:
            print(f"⚠️ Unsupported metric_name: {metric_name}. We want: {LOGLOSS} or {ROC_AUC_OVR}.")
        score = auc
    return Metrics(score=score, metrics=metrics)

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