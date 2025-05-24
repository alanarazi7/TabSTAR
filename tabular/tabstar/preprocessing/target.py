from typing import Tuple

import numpy as np
from pandas import DataFrame

from tabular.datasets.properties import DatasetProperties
from tabular.tabstar.preprocessing.verbalization import verbalize_target_values
from tabular.utils.processing import pd_concat_cols
from tabular.utils.utils import verbose_print


def add_target_tokens(x_txt: DataFrame, x_num: np.ndarray, data: DatasetProperties) -> Tuple[DataFrame, np.ndarray]:
    verbose_print(f"🎯 Adding context and target tokens")
    assert x_txt.shape == x_num.shape
    examples, features = x_txt.shape
    x_num = pad_x_num_with_target_tokens(x_num, data=data)
    x_cat_cols = list(x_txt.columns)
    target_df = create_target_df(data=data, examples=examples)
    x_txt = pd_concat_cols([target_df, x_txt])
    assert list(x_txt.columns) == list(target_df.columns) + x_cat_cols
    assert x_txt.shape == x_num.shape == (examples, features + data.d_effective_output)
    verbose_print(f"🎯 Done!")
    return x_txt, x_num


def pad_x_num_with_target_tokens(x_num: np.ndarray, data: DatasetProperties):
    pad = np.zeros((len(x_num), data.d_effective_output))
    x_num = np.concatenate([pad, x_num], axis=1)
    return x_num


def create_target_df(data: DatasetProperties, examples: int) -> DataFrame:
    target_strings = verbalize_target_values(data)
    target_cols = [f"TABULAR_TARGET_SPECIAL_TOKEN_{i}" for i in range(len(target_strings))]
    target_df = DataFrame({col: [t] * examples for col, t in zip(target_cols, target_strings)})
    return target_df