import pandas as pd
from pandas import DataFrame, Series

from tabstar.preprocessing.texts import sanitize_text


def verbalize_textual_features(x: DataFrame) -> DataFrame:
    for col, dtype in x.dtypes.items():
        if dtype not in {'str', 'float'}:
            raise TypeError(f"Column {col} has unsupported dtype {dtype}. Expected str or float for verbalization.")
        if dtype == 'str':
            x[col] = x[col].apply(lambda v: verbalize_feature(col=str(col), value=v))
    return x

def verbalize_feature(col: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected string value, got {type(value)}")
    v = f"Predictive Feature: {col}\nFeature Value: {value}"
    v = sanitize_text(v)
    return v

def prepend_target_tokens(x: DataFrame, y: Series, is_cls: bool) -> DataFrame:
    if is_cls:
        y_values = sorted(set(y))
        tokens = [f"Target Feature: {y.name}\nFeature Value: {v}" for v in y_values]
    else:
        tokens = [f"Numerical Target Feature: {y.name}"]
    tokens = [sanitize_text(token) for token in tokens]
    target_df = DataFrame({f"TABSTAR_TARGET_TOKEN_{i}": [t] * len(y) for i, t in enumerate(tokens)}, index=x.index)
    x = pd.concat([target_df, x], axis=1)
    return x
