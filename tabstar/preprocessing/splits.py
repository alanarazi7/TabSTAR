from typing import Tuple

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.constants import SEED

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000
VAL_RATIO = 0.1
MAX_VAL_SIZE = 1000

def split_to_test(x: DataFrame, y: Series, is_cls: bool) -> Tuple[DataFrame, DataFrame, Series, Series]:
    test_size = int(len(y) * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    x_train, x_test, y_train, y_test = do_split(x=x, y=y, test_size=test_size, is_cls=is_cls)
    return x_train, x_test, y_train, y_test

def split_to_val(x: DataFrame, y: Series, is_cls: bool) -> Tuple[DataFrame, DataFrame, Series, Series]:
    val_size = int(len(y) * VAL_RATIO)
    val_size = min(val_size, MAX_VAL_SIZE)
    x_train, x_val, y_train, y_val = do_split(x=x, y=y, test_size=val_size, is_cls=is_cls)
    return x_train, x_val, y_train, y_val


def do_split(x: DataFrame, y: Series, test_size: float, is_cls: bool) -> Tuple[DataFrame, DataFrame, Series, Series]:
    stratify = y if is_cls else None
    try:
        return train_test_split(x, y, test_size=test_size, random_state=SEED, stratify=stratify)
    except ValueError:
        # If stratification fails, fallback to random split
        return train_test_split(x, y, test_size=test_size, random_state=SEED)