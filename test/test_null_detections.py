import numpy as np
import pandas as pd
from pandas import Series

from tabstar.preprocessing.nulls import get_invalid_indices


def test_get_valid_indices():
    s = Series([1, 2, 3, np.nan, np.inf, 6])
    assert get_invalid_indices(s) == {3, 4}

    s = Series([None, 1, float("inf"), np.NaN])
    assert get_invalid_indices(s) == {0, 2, 3}

    s = [1, 2, None, np.nan, pd.NA, float("nan"), float("NaN"), float("inf"), np.inf, -np.inf]
    assert get_invalid_indices(s) == {2, 3, 4, 5, 6, 7, 8, 9}

