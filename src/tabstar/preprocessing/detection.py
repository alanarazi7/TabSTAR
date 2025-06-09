from typing import Any

from pandas import Series

from tabstar.preprocessing.nulls import get_valid_values

MAX_NUMERIC_FOR_CATEGORICAL = 50


def is_mostly_numerical(s: Series) -> bool:
    values = get_valid_values(s)
    unique = set(values)
    n_unique = len(unique)
    if n_unique <= MAX_NUMERIC_FOR_CATEGORICAL:
        return False
    non_numerical_unique = [v for v in unique if not is_numeric(v)]
    if len(non_numerical_unique) > 1:
        return False
    return True



def is_numeric(f: Any) -> bool:
    if f is None:
        return False
    if isinstance(f, str):
        return f.isdigit()
    if isinstance(f, (int, float,)):
        return True
    try:
        f = float(f)
        return True
    except ValueError:
        print(f"ValueError: {f} from type {f} cannot be converted to float")
        return False