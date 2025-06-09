from typing import List, Optional

import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype

from tabstar.preprocessing.texts import sanitize_text


def verbalize_textual_features(x: DataFrame) -> DataFrame:
    semantic_cols = [col for col, dtype in x.dtypes.items() if not is_numeric_dtype(dtype)]
    for col in semantic_cols:
        x[col] = x[col].apply(lambda v: verbalize_feature(col=str(col), value=v))
    return x

def verbalize_feature(col: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected string value, got {type(value)}")
    v = f"Predictive Feature: {col}\nFeature Value: {value}"
    v = sanitize_text(v)
    return v

def prepend_target_tokens(x: DataFrame, y_name: str, y_values: Optional[List[str]]) -> DataFrame:
    if y_values:
        tokens = [f"Target Feature: {y_name}\nFeature Value: {v}" for v in y_values]
    else:
        tokens = [f"Numerical Target Feature: {y_name}"]
    tokens = [sanitize_text(token) for token in tokens]
    target_df = DataFrame({f"TABSTAR_TARGET_TOKEN_{i}": [t] * len(x) for i, t in enumerate(tokens)}, index=x.index)
    x = pd.concat([target_df, x], axis=1)
    return x
