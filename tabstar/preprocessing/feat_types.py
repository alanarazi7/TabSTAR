import pandas as pd
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_bool_dtype

from tabstar.preprocessing.dates import series_to_dt
from tabstar.preprocessing.detection import is_mostly_numerical, is_numeric
from tabstar.preprocessing.nulls import convert_numeric_with_missing, MISSING_VALUE


# TODO: for future versions, maybe best to rely on maintained packages, e.g. skrub's TableVectorizer
def detect_feature_types(x: DataFrame) -> DataFrame:
    for col, dtype in x.dtypes.items():
        col_name = str(col)
        s = x[col_name]
        if is_datetime64_any_dtype(dtype):
            x[col_name] = series_to_dt(s=s)
        elif is_numeric_dtype(dtype) or is_mostly_numerical(s=s):
            x[col_name] = convert_series_to_numeric(s=s)
        elif is_object_dtype(dtype) or is_bool_dtype(dtype):
            x[col_name] = s.astype(object).fillna(MISSING_VALUE).astype(str)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}")
    return x

def convert_series_to_numeric(s: Series) -> Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    non_numeric_indices = [not is_numeric(f) for f in s]
    if not any(non_numeric_indices):
        return s.astype(float)
    unique_non_numeric = s[non_numeric_indices].unique()
    if len(unique_non_numeric) != 1:
        raise ValueError(f"Missing values detected are {unique_non_numeric}. Should be only one!")
    missing_value = unique_non_numeric[0]
    return convert_numeric_with_missing(s=s, missing_value=missing_value)
