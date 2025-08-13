from dataclasses import dataclass
from typing import Union, Dict, Any

import numpy as np
import torch
from numpy.exceptions import AxisError
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, accuracy_score, f1_score, log_loss, \
    recall_score, average_precision_score
from torch import Tensor, softmax
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


@dataclass
class Metrics:
    score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        self.score = float(self.score)
        self.metrics = {k: float(v) for k, v in self.metrics.items()}


def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int, is_pretrain: bool = False,
                     is_multilabel: bool = False) -> Metrics:
    if is_multilabel:
        return _calculate_metrics_for_multilabel(y_true=y_true, y_pred=y_pred)
    if d_output == 1:
        return _calculate_metrics_for_regression(y_true=y_true, y_pred=y_pred, is_pretrain=is_pretrain)
    elif d_output == 2:
        return _calculate_metrics_for_binary(y_true=y_true, y_pred=y_pred)
    elif d_output > 2:
        return _calculate_metrics_for_multiclass(y_true=y_true, y_pred=y_pred)
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")

def _calculate_metrics_for_regression(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, is_pretrain: bool) -> Metrics:
    rsq = r2_score(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(mse)
    one_minus_mse = 1 - mse
    metrics = {'r2': rsq, 'mse': mse, '1-mse': one_minus_mse, 'rmse': rmse}
    score = one_minus_mse if is_pretrain else rsq
    return Metrics(score=score, metrics=metrics)

def _calculate_metrics_for_binary(y_true: Union[np.ndarray, Series], y_pred: np.ndarray) -> Metrics:
    y_pred_label = (y_pred > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    logloss = log_loss(y_true, y_pred)
    # TODO: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
    # precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    metrics = {'auc': auc, 'accuracy': acc, 'f1': f1, 'logloss': logloss,
               # 'precision': precision,
               'recall': recall}
    return Metrics(score=auc, metrics=metrics)

def _calculate_metrics_for_multiclass(y_true: Union[np.ndarray, Series], y_pred: np.ndarray) -> Metrics:
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
    # TODO: UndefinedMetricWarning: 'k' (3) greater than or equal to 'n_classes' (3) will result in a perfect score and is therefore meaningless
    # try:
    #     topk_acc = top_k_accuracy_score(y_true, y_pred, k=3, labels=np.unique(y_true))
    # except Exception:
    #     topk_acc = np.nan
    metrics = {
        'auc': auc,
        'logloss': logloss,
        'macro_f1': f1,
        'accuracy': acc,
        # f'top3_accuracy': topk_acc,
    }
    return Metrics(score=auc, metrics=metrics)

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


def apply_loss_fn(prediction: Tensor, d_output: int, is_multilabel: bool = False) -> Tensor:
    if is_multilabel:
        return prediction.sigmoid()
    if d_output == 1:
        return prediction
    prediction = prediction.to(torch.float32)
    prediction = softmax(prediction, dim=1)
    if d_output == 2:
        # We want the probability of '1'
        prediction = prediction[:, 1]
    return prediction


def calculate_loss(predictions: Tensor, y: Union[Series, np.ndarray], d_output: int,
                   is_multilabel: bool = False) -> Tensor:
    is_reg = bool(d_output == 1)
    if is_reg:
        loss_fn = MSELoss()
        dtype = torch.float32
    elif is_multilabel:
        loss_fn = BCEWithLogitsLoss()
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


def _calculate_metrics_for_multilabel(
    y_true: Union[np.ndarray, Series],
    y_pred: np.ndarray,
    threshold: float = 0.5
):
    """
    y_true: (B, C) in {0,1}
    y_pred: (B, C) probabilities in [0,1]
    """
    y_true = np.array(list(y_true), dtype=int)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    assert np.all((0.0 <= y_pred) & (y_pred <= 1.0)), "y_pred must be in [0, 1]"

    y_hat = (y_pred >= threshold).astype(int)

    ap_micro = float(average_precision_score(y_true, y_pred, average="micro"))
    f1_micro = float(f1_score(y_true, y_hat, average="micro", zero_division=0))

    score = ap_micro if not np.isnan(ap_micro) else f1_micro
    return Metrics(score=score, metrics={"ap_micro": ap_micro, "f1_micro": f1_micro})