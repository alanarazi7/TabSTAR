from typing import Dict, Set

import torch
from pandas import DataFrame, Series
from skrub import TextEncoder

from tabstar.arch.config import E5_SMALL
from tabstar.preprocessing.nulls import get_valid_values

MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100


def fit_text_encoders(x: DataFrame, numerical_features: Set[str], device: torch.device) -> Dict[str, TextEncoder]:
    text_encoders = {}
    for col, dtype in x.dtypes.items():
        if (col not in numerical_features) and _is_text_feature(s=x[col]):
            text_encoders[str(col)] = TextEncoder(model_name=E5_SMALL, device=device)
    return text_encoders


def _is_text_feature(s: Series) -> bool:
    values = get_valid_values(s)
    if not values:
        return False
    n_unique = len(set(values))
    if n_unique >= MIN_TEXT_UNIQUE_FREQUENCY:
        return True
    unique_ratio = n_unique / len(values)
    return unique_ratio >= MIN_TEXT_UNIQUE_RATIO
