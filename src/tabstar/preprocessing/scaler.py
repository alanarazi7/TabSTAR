from pandas import Series
from sklearn.preprocessing import StandardScaler

from tabstar.preprocessing.nulls import get_invalid_indices

Z_MAX_ABS_VAL = 3


def fit_standard_scaler(s: Series) -> StandardScaler:
    s = s.copy().dropna()
    scaler = StandardScaler()
    scaler.fit(s.values.reshape(-1, 1))
    return scaler

def transform_clipped_z_scores(s: Series, scaler: StandardScaler, allow_null: bool = False) -> Series:
    invalid = get_invalid_indices(s)
    s = s.copy()
    if allow_null:
        s = s.fillna(s.mean())
    s_val = scaler.transform(s.values.reshape(-1, 1)).flatten()
    s_val = s_val.clip(-Z_MAX_ABS_VAL, Z_MAX_ABS_VAL)
    if invalid:
        s_val[sorted(invalid)] = 0
    s = Series(s_val, index=s.index, name=s.name)
    return s