from typing import Tuple

from pandas import DataFrame, Series, SparseDtype


def densify_objects(x: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    if not (isinstance(x, DataFrame) and isinstance(y, Series)):
        raise ValueError(f"Expected DataFrame and Series, got {type(x)} and {type(y)}")
    for col in x.columns:
        x[col] = densify_series(x[col])
    y = densify_series(y)
    return x, y



def densify_series(s: Series) -> Series:
    if not isinstance(s.dtype, SparseDtype):
        return s
    return s.sparse.to_dense()