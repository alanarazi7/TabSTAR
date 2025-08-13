from typing import Union

import numpy as np
from pandas import Series
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer

from tabstar.preprocessing.scaler import fit_standard_scaler, transform_clipped_z_scores


def transform_preprocess_y(y: Union[Series, np.ndarray], scaler: Union[LabelEncoder, StandardScaler]) -> Series:
    y = y.copy()
    if isinstance(scaler, StandardScaler):
        return transform_clipped_z_scores(s=y, scaler=scaler)
    elif isinstance(scaler, LabelEncoder):
        return transform_cls_y(y=y, encoder=scaler)
    elif isinstance(scaler, MultiLabelBinarizer):
        return transform_multilabel_y(y=y, encoder=scaler)
    raise TypeError(f"What is this scaler {scaler} from type {type(scaler)}?")


def transform_cls_y(y: Series, encoder: LabelEncoder) -> Series:
    y_val = encoder.transform(y)
    return Series(y_val, name=y.name, index=y.index)


def fit_preprocess_y(y: Series, is_cls: bool, is_multilabel: bool = False) -> Union[LabelEncoder, StandardScaler, MultiLabelBinarizer]:
    if is_multilabel:
        return fit_multilabel_y(y)
    elif is_cls:
        return fit_cls_y(y)
    else:
        return fit_reg_y(y)

def fit_cls_y(y: Series) -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder

def fit_reg_y(y: Series) -> StandardScaler:
    return fit_standard_scaler(s=y)

def fit_multilabel_y(y: Series) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    return mlb


def transform_multilabel_y(y: Series, encoder: MultiLabelBinarizer) -> Series:
    y_val = encoder.transform(y)
    y_val_tuples = list(map(tuple, y_val))
    return Series(y_val_tuples, name=y.name, index=y.index)  # Convert to tuples for consistency