from typing import Optional

import pandas as pd
from pandas import DataFrame
from skrub import DatetimeEncoder

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.objects import FeatureType
from tabular.utils.utils import verbose_print


def process_dates(raw: RawDataset):
    date_features = list(raw.dates)
    for col in date_features:
        date_df = DatetimeEncoder(add_weekday=True, add_total_seconds=True).fit_transform(raw.x[col])
        date_df = _filter_dt_features(raw=raw, date_df=date_df)
        assert date_df is not None, f"❌ No date features to include for {col}"
        raw.x.drop(columns=[col], inplace=True)
        raw.x = pd.concat([raw.x, date_df], axis=1)
        raw.feature_types[FeatureType.DATE].remove(col)


def _filter_dt_features(raw: RawDataset, date_df: DataFrame) -> Optional[DataFrame]:
    cols_to_include = []
    for new_col in date_df.columns:
        if len(set(date_df[new_col])) < 2:
            continue
        cols_to_include.append(new_col)
        raw.feature_types[FeatureType.NUMERIC].add(new_col)
        d_summary = date_df[new_col].describe().to_dict()
        d_summary = {k: round(v, 2) for k, v in d_summary.items() if k in ['min', 'max', 'mean']}
        verbose_print(f"➕ Including {new_col} in the dataset: {d_summary}")
    if not cols_to_include:
        return None
    return date_df[cols_to_include]
