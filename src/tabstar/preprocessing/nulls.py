from typing import List, Any, Optional, Set

import numpy as np
import pandas as pd
from pandas import Series

MISSING_VALUE = "Unknown Value"


def raise_if_null_target(y: Series):
    # TODO: do we want to allow throwing these away with a flag?
    # TODO: for multiclass, warn if any category is too rare?
    missing = y.isnull()
    y_missing = missing.sum()
    if y_missing > 0:
        raise ValueError(f"Target variable {y.name} has {y_missing} null values, please handle them before training.")


def get_invalid_indices(ls: Series) -> Set[int]:
    return {i for i, x in enumerate(ls) if _get_non_null_value(x) is None}


def get_valid_values(ls: Series) -> List:
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


def convert_numeric_with_missing(s: pd.Series, missing_value: Any) -> pd.Series:
    return s.apply(lambda x: x if x != missing_value else np.nan).astype(float)
