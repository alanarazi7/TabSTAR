from typing import List

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def transform_encoded(col: Series, values: np.ndarray, is_float: bool = False) -> Series:
    if is_float:
        dtype = np.float32
    else:
        dtype = np.int64
    return Series(values.astype(dtype), name=col.name, index=col.index)


def pd_concat_cols(dataframes: List[DataFrame]) -> DataFrame:
    dataframes = [df.reset_index(drop=True) for df in dataframes]
    return pd.concat(dataframes, axis=1)


def pd_indices_to_array(df: DataFrame | Series, indices: List[int]) -> DataFrame | Series:
    return df.iloc[indices].reset_index(drop=True)
