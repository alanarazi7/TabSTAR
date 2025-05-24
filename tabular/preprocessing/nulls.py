from typing import Any, Optional, List, Set

import numpy as np
import pandas as pd
from pandas import Series

MISSING_VALUE = "Unknown Value"

def convert_series_to_numeric(s: pd.Series, missing_value: str = None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    if missing_value is not None:
        return _convert_numeric_with_missing(s=s, missing_value=missing_value)
    non_numeric_indices = [_is_non_numeric(f) for f in s]
    if not any(non_numeric_indices):
        return s.astype(float)
    unique_non_numeric = s[non_numeric_indices].unique()
    if len(unique_non_numeric) == 1:
        missing_value = unique_non_numeric[0]
        return _convert_numeric_with_missing(s=s, missing_value=missing_value)
    raise ValueError(f"Missing values detected are {unique_non_numeric}. Should be only one!")

def get_invalid_indices(ls: List | Series) -> Set[int]:
    return {i for i, x in enumerate(ls) if _get_non_null_value(x) is None}


def get_valid_values(ls: List | Series) -> List:
    return [x for x in ls if _get_non_null_value(x) is not None]

def _get_non_null_value(x: Any) -> Optional[Any]:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return None
    if not np.isfinite(x):
        return None
    if pd.isnull(x):
        return None
    return x

def _convert_numeric_with_missing(s: pd.Series, missing_value: str) -> pd.Series:
    return s.apply(lambda x: x if x != missing_value else np.nan).astype(float)

def _is_non_numeric(f: Any) -> bool:
    if f is None:
        return True
    if isinstance(f, str):
        return not f.isdigit()
    if isinstance(f, (int, float,)):
        return False
    try:
        f = float(f)
        return False
    except ValueError:
        print(f"ValueError: {f} from type {f} cannot be converted to float")
        return True