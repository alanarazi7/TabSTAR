import numpy as np
from pandas import Series
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from tabular.preprocessing.nulls import get_valid_values

def fit_scaler(train_col: Series, for_verbalize: bool) -> QuantileTransformer | StandardScaler:
    train_values = np.array(get_valid_values(train_col))
    if len(train_values) == 0:
        raise ValueError(f"Cannot fit scaler on empty column {train_col.name}, the series values are: {train_col}")
    if for_verbalize:
        scaler = QuantileTransformer(output_distribution='uniform',
                                     n_quantiles=min(1000, len(train_values)),
                                     subsample=1000000000,
                                     random_state=0)
    else:
        scaler = StandardScaler()
    scaler.fit(train_values.reshape(-1, 1))
    return scaler


def get_quantile_levels(num_bins: int) -> np.array:
    # Create equally spaced quantile levels from 0 to 1
    return np.linspace(0, 1, num_bins + 1)


def is_numerical(v: float | str | int) -> bool:
    if isinstance(v, str):
        return v.isdigit()
    elif isinstance(v, (int, float)):
        return True
    raise ValueError(f"Unexpected type {type(v)}: {v}")