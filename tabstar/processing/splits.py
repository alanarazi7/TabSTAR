from typing import Tuple

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.constants import SEED

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000
VAL_RATIO = 0.1
MAX_VAL_SIZE = 1000

# TODO: 'stratify' only for classification? also, for edge cases with stratify, use try except
def split_to_test(x: DataFrame, y: Series) -> Tuple[DataFrame, DataFrame, Series, Series]:
    n = len(y)
    test_size = int(n * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=SEED, stratify=y)
    return x_train, x_test, y_train, y_test

def split_to_val(x: DataFrame, y: Series) -> Tuple[DataFrame, DataFrame, Series, Series]:
    n = len(y)
    val_size = int(n * VAL_RATIO)
    val_size = min(val_size, MAX_VAL_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=SEED, stratify=y)
    return x_train, x_val, y_train, y_val
