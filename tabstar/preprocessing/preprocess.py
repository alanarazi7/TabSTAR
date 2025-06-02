from typing import Tuple

from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype, is_object_dtype

from tabstar.preprocessing.dates import preprocess_dates
from tabstar.preprocessing.feat_types import detect_feature_types
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.sparse import densify_objects
from tabstar.preprocessing.texts import replace_column_names
from tabstar.preprocessing.verbalize import verbalize_textual_features, prepend_target_tokens


def preprocess_raw(x: DataFrame, y: Series, is_cls: bool) -> Tuple[DataFrame, Series]:
    raise_if_null_target(y)
    if len(set(x.columns)) != len(x.columns):
        raise ValueError("Duplicate column names found in DataFrame!")
    x, y = densify_objects(x=x, y=y)
    x = detect_feature_types(x=x)
    x = preprocess_dates(x=x)
    x, y = replace_column_names(x=x, y=y)
    x = verbalize_textual_features(x=x)
    x = prepend_target_tokens(x=x, y=y, is_cls=is_cls)
    _assert_final_dtypes(x)
    return x, y


def _assert_final_dtypes(x: DataFrame):
    for col, dtype in x.dtypes.items():
        if is_numeric_dtype(dtype) or is_object_dtype(dtype):
            continue
        raise TypeError(f"Column {col} has unsupported dtype {dtype}")