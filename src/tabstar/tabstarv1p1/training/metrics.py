from dataclasses import dataclass
from typing import Dict

import numpy as np
from pandas import Series
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

from tabstar.tabstarv1p1.training.loss import multiclass_roc_auc_score


@dataclass
class Metrics:
    score: float
    metrics: Dict[str, float]

    def __post_init__(self):
        self.score = float(self.score)
        self.metrics = {k: float(v) for k, v in self.metrics.items()}


def calculate_metric(y_true: np.ndarray | Series, y_pred: np.ndarray, d_output: int) -> Metrics:
    if d_output == 1:
        rsq = r2_score(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        return Metrics(score=rsq, metrics={"r2": rsq, "mse": mse, "1-mse": 1 - mse})
    elif d_output == 2:
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        return Metrics(score=auc, metrics={"roc_auc": auc})
    elif d_output > 2:
        auc = multiclass_roc_auc_score(y_true=y_true, y_pred=y_pred)
        return Metrics(score=auc, metrics={"roc_auc_ovr": auc})
    raise ValueError(f"Unsupported d_output: {d_output}. Expected 1 (regression), 2 (binary), or >2 (multiclass).")
