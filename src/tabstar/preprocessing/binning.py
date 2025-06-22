from typing import List

import numpy as np
from pandas import Series
from sklearn.preprocessing import QuantileTransformer

from tabstar.preprocessing.nulls import get_invalid_indices, MISSING_VALUE
from tabstar.preprocessing.verbalize import verbalize_feature

VERBALIZED_QUANTILE_BINS = 10

def fit_numerical_bins(s: Series) -> QuantileTransformer:
    s = s.copy().dropna()
    scaler = QuantileTransformer(output_distribution='uniform',
                                 n_quantiles=min(1000, len(s)),
                                 subsample=1000000000,
                                 random_state=0)
    scaler.fit(s.values.reshape(-1, 1))
    return scaler


def transform_numerical_bins(s: Series, scaler: QuantileTransformer) -> Series:
    invalid_indices = get_invalid_indices(s)
    quantile_levels = np.linspace(0, 1, VERBALIZED_QUANTILE_BINS + 1)
    boundaries = scaler.inverse_transform(quantile_levels.reshape(-1, 1)).flatten()
    assert len(boundaries) == VERBALIZED_QUANTILE_BINS + 1
    verbalized_bins = verbalize_bins(boundaries)
    bin_index = np.digitize(s, boundaries)
    verbalized = [verbalized_bins[i] for i in bin_index]
    for idx in invalid_indices:
        verbalized[idx] = MISSING_VALUE
    verbalized = [verbalize_feature(col=str(s.name), value=v) for v in verbalized]
    s = Series(verbalized, index=s.index, name=s.name)
    return s


def verbalize_bins(boundaries: np.array) -> List[str]:
    # TODO: this can become a bit ugly with high-precision numbers, or relatively-discrete numerical values
    boundaries = [format_float(b) for b in boundaries]
    first = f"Lower than {min(boundaries)} (Quantile 0%)"
    last = f"Higher than {max(boundaries)} (Quantile 100%)"
    bins = []
    for i, b in enumerate(boundaries[:-1]):
        r = f"{b} to {boundaries[i + 1]}"
        low = i * VERBALIZED_QUANTILE_BINS
        high = (i + 1) * VERBALIZED_QUANTILE_BINS
        q = f"(Quantile {low} - {high}%)"
        bins.append(f"{r} {q}")
    assert len(bins) == VERBALIZED_QUANTILE_BINS == len(boundaries) - 1
    bins = [first] + bins + [last]
    return bins


def format_float(num: float) -> str:
    rounded = round(num, 4)
    if rounded.is_integer():
        return str(int(rounded))
    formatted = f"{rounded:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted