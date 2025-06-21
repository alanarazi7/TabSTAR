from typing import Tuple, Optional

import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.constants import SEED

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000
VAL_RATIO = 0.1
MAX_VAL_SIZE = 1000

def split_to_test(x: DataFrame, y: Series, is_cls: bool, seed: int = SEED, train_examples: Optional[int] = None) -> Tuple[DataFrame, DataFrame, Series, Series]:
    test_size = int(len(y) * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    if (train_examples is not None) and len(x) > train_examples:
        test_size = len(x) - train_examples
    x_train, x_test, y_train, y_test = do_split(x=x, y=y, test_size=test_size, is_cls=is_cls, seed=seed)
    return x_train, x_test, y_train, y_test

def split_to_val(x: DataFrame, y: Series, is_cls: bool, seed: int = SEED, val_ratio: float = VAL_RATIO) -> Tuple[DataFrame, DataFrame, Series, Series]:
    val_size = int(len(y) * val_ratio)
    val_size = min(val_size, MAX_VAL_SIZE)
    x_train, x_val, y_train, y_val = do_split(x=x, y=y, test_size=val_size, is_cls=is_cls, seed=seed)
    return x_train, x_val, y_train, y_val


def do_split(x: DataFrame, y: Series, test_size: int, is_cls: bool, seed: int) -> Tuple[DataFrame, DataFrame, Series, Series]:
    if not is_cls:
        return train_test_split(x, y, test_size=test_size, random_state=seed)
    has_rare_class = y.value_counts().min() == 1
    if has_rare_class:
        return _split_with_rare_classes(x, y, test_size=test_size, seed=seed)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def _split_with_rare_classes(x: DataFrame, y: Series, test_size: float, seed: int) -> Tuple[DataFrame, DataFrame, Series, Series]:
    # TODO: add tests here, seems like a complex function
    singleton_classes = y.value_counts()[y.value_counts() == 1].index
    is_singleton = y.isin(singleton_classes)
    x_single = x[is_singleton]
    y_single = y[is_singleton]
    x_rest = x[~is_singleton]
    y_rest = y[~is_singleton]

    x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=test_size, random_state=seed,
                                                        stratify=y_rest)

    # Add singletons to train, shuffle
    x_train = pd.concat([x_train, x_single])
    y_train = pd.concat([y_train, y_single])
    shuffled = x_train.sample(frac=1, random_state=seed).index

    x_train = x_train.loc[shuffled]
    y_train = y_train.loc[shuffled]

    return x_train, x_test, y_train, y_test
