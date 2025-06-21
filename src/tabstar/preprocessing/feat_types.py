from typing import Set, Optional

import pandas as pd
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_bool_dtype

from tabstar.preprocessing.detection import is_mostly_numerical, is_numeric
from tabstar.preprocessing.nulls import convert_numeric_with_missing, MISSING_VALUE, get_valid_values


def transform_feature_types(x: DataFrame, numerical_features: Set[str]) -> DataFrame:
    for col in x.columns:
        if col in numerical_features:
            x[col] = convert_series_to_numeric(s=x[col])
        else:
            x[col] = x[col].astype(object).fillna(MISSING_VALUE).astype(str)
    return x


def detect_numerical_features(x: DataFrame) -> Set[str]:
    return {col for col in x.columns if is_numerical_feature(s=x[col])}

# TODO: for future versions, maybe best to rely on maintained packages, e.g. skrub's TableVectorizer
def is_numerical_feature(s: Series) -> bool:
    if is_datetime64_any_dtype(s.dtype):
        raise TypeError(f"At this point, dates should have already been transformed.")
    elif len(get_valid_values(s)) == 0:
        return False
    elif is_numeric_dtype(s.dtype) or is_mostly_numerical(s=s):
        return True
    elif is_object_dtype(s.dtype) or is_bool_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return False
    else:
        raise ValueError(f"Unsupported dtype {s.dtype} for series {s.name}")


def convert_series_to_numeric(s: Series, missing_value: Optional[str] = None) -> Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    non_numeric_indices = [not is_numeric(f) for f in s]
    if not any(non_numeric_indices):
        return s.astype(float)
    if missing_value is None:
        unique_non_numeric = s[non_numeric_indices].unique()
        if len(unique_non_numeric) != 1:
            raise ValueError(f"Missing values detected are {unique_non_numeric}. Should be only one!")
        missing_value = unique_non_numeric[0]
    return convert_numeric_with_missing(s=s, missing_value=missing_value)