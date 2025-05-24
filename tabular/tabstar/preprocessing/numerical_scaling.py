import numpy as np
from pandas import Series
from sklearn.preprocessing import StandardScaler

from tabular.preprocessing.nulls import get_invalid_indices
from tabular.preprocessing.target import Z_MAX_ABS_VAL


def standardize_series(series: Series, scaler: StandardScaler) -> np.ndarray:
    # Detecting missing values, remembering their index and temporarily filling with median
    invalid_indices = get_invalid_indices(series)
    series = series.fillna(series.median())
    np_series = series.to_numpy()
    # Scaling the series and clipping the value
    scaled = scaler.transform(np_series.reshape(-1, 1))
    scaled = np.clip(scaled, -Z_MAX_ABS_VAL, Z_MAX_ABS_VAL)
    scaled = scaled[:, 0].tolist()
    for idx in invalid_indices:
        scaled[idx] = 0
    scaled = np.array(scaled)
    return scaled
