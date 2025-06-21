from typing import Any, Dict

import pandas as pd
from pandas import Series, DataFrame
from pandas.core.dtypes.common import is_datetime64_any_dtype
from skrub import DatetimeEncoder


def transform_date_features(x: DataFrame, date_transformers: Dict[str, DatetimeEncoder]) -> DataFrame:
    rows = x.shape[0]
    for col, dt_encoder in date_transformers.items():
        s = series_to_dt(s=x[col])
        dt_df = dt_encoder.transform(s)
        dt_df.index = x.index
        x = x.drop(columns=[col])
        x = pd.concat([x, dt_df], axis=1)
        if x.shape[0] != rows:
            raise ValueError(f"Row mismatch after transforming date column {col}: expected {rows}, got {x.shape}")
    return x

def fit_date_encoders(x: DataFrame) -> Dict[str, DatetimeEncoder]:
    date_encoders = {}
    date_columns = [str(col) for col, dtype in x.dtypes.items() if is_datetime64_any_dtype(dtype)]
    for col in date_columns:
        dt_s = series_to_dt(s=x[col])
        # Adds: "year", "month", "day", "hour", "total_seconds", "weekday"
        encoder = DatetimeEncoder(add_weekday=True, add_total_seconds=True)
        encoder.fit(dt_s)
        date_encoders[col] = encoder
    return date_encoders

def series_to_dt(s: Series) -> Series:
    # TODO: do we want to handle missing values here?
    s = s.apply(_clean_dirty_date)
    dt_s = pd.to_datetime(s, errors='coerce')
    dt_s = dt_s.apply(_remove_timezone)
    return dt_s


def _remove_timezone(dt):
    if pd.notnull(dt) and getattr(dt, 'tzinfo', None) is not None:
        return dt.tz_localize(None)
    return dt


def _clean_dirty_date(s: Any) -> Any:
    if isinstance(s, str):
        s = s.replace('"', "")
    return s