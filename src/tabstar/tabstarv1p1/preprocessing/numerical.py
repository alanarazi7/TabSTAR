import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

# TODO: add / test: tanh_z, RobustScaler, log1p_std, power, sigmoid,
# - power transform (sign(x) * sqrt(abs(x)))
# - log1p standardized: np.log1p(np.clip(safe_values, a_min=0, a_max=None)); self.log1p_scaler.transform(log1p_val)
# - robust: RobustScaler(); self.robust.transform(safe_values)

Z_MAX_ABS = 3
MINMAX_MARGIN = 5

class RichNumericalTransformer:

    def __init__(self, d_num: int):
        self.d_num = d_num
        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler()
        self.quantile = QuantileTransformer(output_distribution='uniform')

    @classmethod
    def fit_series(cls, s: pd.Series, d_num: int) -> "RichNumericalTransformer":
        transformer = RichNumericalTransformer(d_num=d_num)
        transformer.fit(s)
        return transformer

    def fit(self, s: pd.Series):
        values = s.values.reshape(-1, 1)
        valid = ~np.isnan(values).flatten()
        safe_values = values[valid]
        self.scaler.fit(safe_values)
        self.minmax.fit(safe_values)
        with warnings.catch_warnings():
            # UserWarning: n_quantiles (1000) is greater than the total number of samples
            warnings.simplefilter("ignore", category=UserWarning)
            self.quantile.fit(safe_values)
        return self

    def transform(self, s: pd.Series):
        values = s.values.reshape(-1, 1)
        nan_mask = np.isnan(values).flatten()
        # Impute for transform, but keep track of missing for the flag
        impute_mean = np.nanmean(values)
        safe_values = np.where(nan_mask, impute_mean, values.flatten()).reshape(-1, 1)
        z, outlier = self.get_clipped_z_and_outliers(safe_values)
        mm = self.minmax.transform(safe_values)
        mm = np.clip(mm, -MINMAX_MARGIN, 1 + MINMAX_MARGIN)
        quant = self.quantile.transform(safe_values)
        missing = nan_mask.astype(float).reshape(-1, 1)
        # Ones vector, to differentiate from non-numerical features
        ones = np.ones_like(values)
        all_vectors = [z, mm, quant, outlier, missing, ones]
        assert len(all_vectors) >= self.d_num, f"{len(all_vectors)} vs {self.d_num}"
        all_vectors = all_vectors[:self.d_num]
        vec = np.hstack(all_vectors)
        assert vec.shape[1] == self.d_num
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def get_clipped_z_and_outliers(self, safe_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_raw = self.scaler.transform(safe_values).flatten()
        # Careful: must compute outliers BEFORE clipping
        outlier = (np.abs(z_raw) > Z_MAX_ABS).astype(float).reshape(-1, 1)
        z = np.clip(z_raw, -Z_MAX_ABS, Z_MAX_ABS).reshape(-1, 1)
        return z, outlier