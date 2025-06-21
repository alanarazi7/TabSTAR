from typing import Union

import numpy as np
import torch
from numpy.exceptions import AxisError
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score
from torch import Tensor, softmax



def calculate_metric(y_true: Union[np.ndarray, Series], y_pred: np.ndarray, d_output: int) -> float:
    if d_output == 1:
        score = r2_score(y_true=y_true, y_pred=y_pred)
    elif d_output == 2:
        score = roc_auc_score(y_true=y_true, y_score=y_pred)
    elif d_output > 2:
        try:
            score = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
        except (ValueError, AxisError) as e:
            # Error calculating AUC, likely due to class imbalance or insufficient samples
            score = per_class_auc(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError(f"Unsupported number of output classes: {d_output}")
    return float(score)


def per_class_auc(y_true, y_pred) -> float:
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
        return prediction
    prediction = prediction.to(torch.float32)
    prediction = softmax(prediction, dim=1)
    if d_output == 2:
        # We want the probability of '1'
        prediction = prediction[:, 1]
    return prediction