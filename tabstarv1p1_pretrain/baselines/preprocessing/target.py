from typing import Optional

import numpy as np
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from tabstar.tabstarv1p1.preprocessing.target import fit_cls_y, transform_cls_y


def fit_preprocess_y(y: Series, is_cls: bool) -> Optional[LabelEncoder]:
    if not is_cls:
        return None
    return fit_cls_y(y)


def transform_preprocess_y(y: Series | np.ndarray, scaler: Optional[LabelEncoder]) -> Series:
    if scaler is None:
        return y
    assert isinstance(scaler, LabelEncoder), f"Scaler must be LabelEncoder if not None, but got {type(scaler)}"
    y = y.copy()
    return transform_cls_y(y=y, encoder=scaler)
