from dataclasses import dataclass, field
from typing import List

import numpy as np
from pandas import Series
from sklearn.metrics import roc_auc_score, mean_squared_error
from tabpfn_extensions.scoring.scoring_utils import safe_roc_auc_score

from tabular.preprocessing.objects import SupervisedTask

from tabular.utils.utils import verbose_print


def calculate_metric(task_type: SupervisedTask, y_true: Series | np.ndarray, y_pred: Series | np.ndarray) -> float:
    if task_type == SupervisedTask.REGRESSION:
        # 1 - MSE is more stable than R^2 during training
        score  = 1 - mean_squared_error(y_true=y_true, y_pred=y_pred)
        return score
    elif task_type == SupervisedTask.BINARY:
        score = roc_auc_score(y_true=y_true, y_score=y_pred)
    elif task_type == SupervisedTask.MULTICLASS:
        try:
            score = safe_roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr', average='macro')
        except ValueError as e:
            verbose_print(f"⚠️ Error calculating AUC. {y_true=}, {y_pred=}, {e=}")
            score = per_class_auc(y_true=y_true, y_pred=y_pred)
    else:
        raise ValueError(f"Unsupported data properties: {task_type}")
    return float(score)


def per_class_auc(y_true, y_pred) -> float:
    present_classes = np.unique(y_true)
    aucs = {}
    for cls in present_classes:
        # Binary ground truth: 1 for the current class, 0 for others
        y_true_binary = (y_true == cls).astype(int)
        # Predicted probabilities for the current class
        y_pred_scores = y_pred[:, cls]
        try:
            auc = roc_auc_score(y_true_binary, y_pred_scores)
            aucs[cls] = auc
        except ValueError as e:
            verbose_print(f"⚠️ Error calculating AUC for class {cls}. {e=}, {y_true_binary=}, {y_pred_scores=}")
    macro_avg = float(np.mean(list(aucs.values())))
    return macro_avg