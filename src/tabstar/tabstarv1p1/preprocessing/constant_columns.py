from typing import List

from pandas import DataFrame


def detect_constant_columns(x: DataFrame) -> List[str]:
    """Columns with nothing to say: one repeated value, or no value at all."""
    return [str(col) for col in x.columns if x[col].nunique() <= 1]


def drop_constant_columns(x: DataFrame, constant_columns: List[str]) -> DataFrame:
    return x.drop(columns=constant_columns, errors="ignore")
