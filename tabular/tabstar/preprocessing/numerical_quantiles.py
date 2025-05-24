from typing import List

import numpy as np
from pandas import Series
from sklearn.preprocessing import QuantileTransformer

from tabular.preprocessing.nulls import MISSING_VALUE, get_invalid_indices
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.tabstar.preprocessing.numerical_utils import get_quantile_levels

VERBALIZED_QUANTILE_BINS = 10


def verbalized_quantiles_og_values(series: Series, quantile_scaler: QuantileTransformer,
                                   number_verbalization: NumberVerbalization):
    invalid_indices = get_invalid_indices(series)
    verbalized = verbalize_series_into_ranges(series, quantile_scaler=quantile_scaler,
                                              number_verbalization=number_verbalization)
    for idx in invalid_indices:
        verbalized[idx] = MISSING_VALUE
    return verbalized


def verbalize_series_into_ranges(series: Series, quantile_scaler: QuantileTransformer,
                                 number_verbalization: NumberVerbalization) -> List[str]:
    boundaries = get_quantile_boundaries(quantile_scaler)
    assert len(boundaries) == VERBALIZED_QUANTILE_BINS + 1
    verbalized_bins = verbalize_bins(boundaries, number_verbalization)
    bin_index = np.digitize(series, boundaries)
    verbalized = [verbalized_bins[i] for i in bin_index]
    return verbalized


def get_quantile_boundaries(quantile_scaler: QuantileTransformer):
    assert isinstance(quantile_scaler, QuantileTransformer)
    quantile_levels = get_quantile_levels(num_bins=VERBALIZED_QUANTILE_BINS)
    boundaries = quantile_scaler.inverse_transform(quantile_levels.reshape(-1, 1)).flatten()
    return boundaries


def verbalize_bins(boundaries: np.array, number_verbalization: NumberVerbalization) -> List[str]:
    # TODO: this can become a bit ugly with high-precision numbers, or relatively-discrete numerical values
    boundaries = [format_float(b) for b in boundaries]
    quantiles = []
    ranges = []
    for i, b in enumerate(boundaries[:-1]):
        low = i * VERBALIZED_QUANTILE_BINS
        high = (i + 1) * VERBALIZED_QUANTILE_BINS
        quantiles.append(f"(Quantile {low} - {high}%)")
        ranges.append(f"{b} to {boundaries[i + 1]}")
    assert len(ranges) == len(quantiles) == VERBALIZED_QUANTILE_BINS == len(boundaries) - 1
    ranges = [f"Lower than {min(boundaries)}"] + ranges + [f"Higher than {max(boundaries)}"]
    quantiles = ["(Quantile 0%)"] + quantiles + ["(Quantile 100%)"]

    bins = []
    for r, q in zip(ranges, quantiles):
        if number_verbalization == NumberVerbalization.FULL:
            bins.append(f"{r} {q}")
        elif number_verbalization == NumberVerbalization.RANGE:
            bins.append(r)
        elif number_verbalization == NumberVerbalization.NONE:
            bins.append("Numeric")
        else:
            raise ValueError(f"Unsupported number verbalization: {number_verbalization}")
    return bins


def format_float(num: float) -> str:
    rounded = round(num, 4)
    if rounded.is_integer():
        return str(int(rounded))
    formatted = f"{rounded:.4f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted