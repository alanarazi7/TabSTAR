from dataclasses import dataclass, field
from typing import Set, Any, List, Self

from pandas import DataFrame, Series

from tabstar.preprocessing.nulls import get_valid_values


MAX_NUMERIC_FOR_CATEGORICAL = 50

DATE_TYPES = {'datetime64[ns]'}
NUMERIC_TYPES = {'int64', 'int', 'float64', 'float'}
SEMANTIC_TYPES = {'object', 'bool'}
SUPPORTED_TYPES = DATE_TYPES | NUMERIC_TYPES | SEMANTIC_TYPES

@classmethod
class FeatureTypes:
    numerical: Set[str] = field(default_factory=set)
    semantic: Set[str] = field(default_factory=set)
    dates: Set[str] = field(default_factory=set)


# TODO: for future versions, it's best if we try to rely on logics of skrub or similar packages
def get_feature_types(x: DataFrame) -> FeatureTypes:
    feature_types = FeatureTypes()
    for col, dtype in x.dtypes.items():
        col_name = str(col)
        if dtype in DATE_TYPES:
            feature_types.dates.add(col_name)
        elif (dtype in NUMERIC_TYPES) or is_mostly_numerical(s=x[col]):
            feature_types.numerical.add(col_name)
        elif dtype in SEMANTIC_TYPES:
            feature_types.semantic.add(col_name)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}. Convert to any of: {SUPPORTED_TYPES=}")
    return feature_types


def is_mostly_numerical(s: Series) -> bool:
    values = get_valid_values(s)
    unique = set(values)
    n_unique = len(unique)
    if n_unique <= MAX_NUMERIC_FOR_CATEGORICAL:
        return False
    non_numerical_unique = [v for v in unique if not is_numerical(v)]
    if len(non_numerical_unique) > 1:
        return False
    return True


def is_numerical(v: float | str | int) -> bool:
    if isinstance(v, str):
        return v.isdigit()
    elif isinstance(v, (int, float)):
        return True
    raise ValueError(f"Unexpected type {type(v)}: {v}")