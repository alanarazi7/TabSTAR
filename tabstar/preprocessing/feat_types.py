import pandas as pd
from pandas import DataFrame, Series

from tabstar.preprocessing.dates import series_to_dt
from tabstar.preprocessing.detection import is_mostly_numerical, is_numeric
from tabstar.preprocessing.nulls import convert_numeric_with_missing, MISSING_VALUE


DATE_TYPES = {'datetime64[ns]'}
NUMERIC_TYPES = {'int64', 'int', 'float64', 'float'}
SEMANTIC_TYPES = {'object', 'bool'}
SUPPORTED_TYPES = DATE_TYPES | NUMERIC_TYPES | SEMANTIC_TYPES


# TODO: for future versions, it's best if we try to rely on maintained packages, e.g. skrub's TableVectorizer
def detect_feature_types(x: DataFrame) -> DataFrame:
    for col, dtype in x.dtypes.items():
        col_name = str(col)
        s = x[col_name]
        if dtype in DATE_TYPES:
            x[col_name] = series_to_dt(s=s)
        elif (dtype in NUMERIC_TYPES) or is_mostly_numerical(s=s):
            x[col_name] = convert_series_to_numeric(s=s)
        elif dtype in SEMANTIC_TYPES:
            x[col_name] = s.astype(object).fillna(MISSING_VALUE).astype(str)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}. Convert to any of: {SUPPORTED_TYPES=}")
    return x


def convert_series_to_numeric(s: Series) -> Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    non_numeric_indices = [not is_numeric(f) for f in s]
    if not any(non_numeric_indices):
        return s.astype(float)
    unique_non_numeric = s[non_numeric_indices].unique()
    if len(unique_non_numeric) != 1:
        raise ValueError(f"Missing values detected are {unique_non_numeric}. Should be only one!")
    missing_value = unique_non_numeric[0]
    return convert_numeric_with_missing(s=s, missing_value=missing_value)
