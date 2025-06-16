from typing import Dict, Set

import pandas as pd
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
        if col in numerical_features:
            continue
        if not _is_text_feature(s=x[col]):
            continue
        encoder = TextEncoder(model_name=E5_SMALL, device=device)
        encoder.fit(x[col])
        text_encoders[str(col)] = encoder
    return text_encoders

def transform_text_features(x: DataFrame, text_encoders: Dict[str, TextEncoder]) -> DataFrame:
    for text_col, text_encoder in text_encoders.items():
        embedding_df = text_encoder.transform(x[text_col])
        cols_before = len(x.columns)
        x = x.drop(columns=text_col)
        embedding_df = embedding_df.set_index(x.index)
        x = pd.concat([x, embedding_df], axis=1)
        assert len(x.columns) == cols_before + text_encoder.n_components - 1
    return x

def _is_text_feature(s: Series) -> bool:
    values = get_valid_values(s)
    if not values:
        return False
    n_unique = len(set(values))
    if n_unique >= MIN_TEXT_UNIQUE_FREQUENCY:
        return True
    unique_ratio = n_unique / len(values)
    return unique_ratio >= MIN_TEXT_UNIQUE_RATIO
