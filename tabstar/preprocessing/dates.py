from typing import Any

import pandas as pd
from pandas import Series, DataFrame
from pandas.core.dtypes.common import is_datetime64_any_dtype
from skrub import DatetimeEncoder

from tabstar.preprocessing.detection import is_mostly_numerical


def preprocess_dates(x: DataFrame) -> DataFrame:
    date_columns = [col for col, dtype in x.dtypes.items() if is_datetime64_any_dtype(dtype)]
    for col in date_columns:
        dt_df = date_feature_to_df(s=x[col])
        x = x.drop(columns=[col])
        x = pd.concat([x, dt_df], axis=1)
    return x


def date_feature_to_df(s: Series) -> DataFrame:
    date_df = DatetimeEncoder(add_weekday=True, add_total_seconds=True).fit_transform(s)
    date_columns = [c for c in date_df.columns if len(set(date_df[c])) >= 2]
    for col in date_columns:
        dtype = float if is_mostly_numerical(s=date_df[col]) else str
        date_df[col] = date_df[col].astype(dtype)
    if not date_columns:
        raise ValueError(f"No valid date features found in the series: {s}.")
    return date_df[date_columns]

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