from typing import Tuple

from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype


def remove_semantics(x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    # Since the data-synth generator works only with numerical data, we remove the semantics
    for col in x.columns:
        if is_numeric_dtype(x[col]):
            continue
        x[col] = transform_to_categorical(x[col])
    if not is_numeric_dtype(y):
        y = transform_to_categorical(y)
    return x, y


def fill_nulls(x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    # Fill nulls with mean for numerical columns, and with a new category for categorical columns
    for col in x.columns:
        if is_numeric_dtype(x[col]):
            mean_val = x[col].mean()
            x[col] = x[col].fillna(mean_val)
    if is_numeric_dtype(y):
        mean_val = y.mean()
        y = y.fillna(mean_val)
    return x, y


def transform_to_categorical(s: Series):
    val2idx = {}
    for v in s.unique():
        if v not in val2idx:
            val2idx[v] = float(len(val2idx))
    s_cat = s.map(val2idx)
    return s_cat