from typing import Union

from pandas import Series
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tabstar.preprocessing.scaler import fit_standard_scaler, transform_clipped_z_scores


def transform_preprocess_y(y: Series, scaler: Union[LabelEncoder, StandardScaler]) -> Series:
    y = y.copy()
    if isinstance(scaler, StandardScaler):
        return transform_clipped_z_scores(s=y, scaler=scaler)
    elif isinstance(scaler, LabelEncoder):
        return transform_cls_y(y=y, encoder=scaler)
    raise TypeError(f"What is this scaler {scaler} from type {type(scaler)}?")


def transform_cls_y(y: Series, encoder: LabelEncoder) -> Series:
    y_val = encoder.transform(y)
    return Series(y_val, name=y.name, index=y.index)


def fit_preprocess_y(y: Series, is_cls: bool) -> Union[LabelEncoder, StandardScaler]:
    if is_cls:
        return fit_cls_y(y)
    else:
        return fit_reg_y(y)

def fit_cls_y(y: Series) -> LabelEncoder:
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder

def fit_reg_y(y: Series) -> StandardScaler:
    return fit_standard_scaler(s=y)