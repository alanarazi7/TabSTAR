from typing import Tuple, List

import numpy as np
from pandas import DataFrame
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.splits import DataSplit, get_x_train
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.tabstar.preprocessing.numerical_quantiles import verbalized_quantiles_og_values
from tabular.tabstar.preprocessing.numerical_scaling import standardize_series
from tabular.tabstar.preprocessing.numerical_utils import fit_scaler
from tabular.utils.utils import verbose_print


def scale_x_num_and_add_categorical_bins(raw: RawDataset, splits: List[DataSplit],
                                         number_verbalization: NumberVerbalization) -> Tuple[DataFrame, np.ndarray]:
    verbose_print(f"Scaling {len(raw.numerical)} numerical features for {raw.sid}")
    x_num = np.zeros(shape=raw.x.shape, dtype=np.float32)
    sorted_cols = raw.numerical + raw.bool_cat_text
    x_train = get_x_train(x=raw.x, splits=splits)
    for i, col in enumerate(raw.numerical):
        train_col = x_train[col]
        quantile_scaler = fit_scaler(train_col, for_verbalize=True)
        original_col = raw.x[col].copy()
        raw.x[col] = verbalized_quantiles_og_values(series=original_col, quantile_scaler=quantile_scaler,
                                                    number_verbalization=number_verbalization)
        standard_scaler = fit_scaler(train_col, for_verbalize=False)
        scaled = standardize_series(series=original_col, scaler=standard_scaler)
        x_num[:, i] = scaled
    x_txt = raw.x[sorted_cols].copy()
    verbose_print(f"Done scaling numerical!")
    return x_txt, x_num