import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.constants import SEED

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000
VAL_RATIO = 0.1
MAX_VAL_SIZE = 1000

def split_to_test(x: DataFrame, y: Series, is_cls: bool, fold: int = -1, train_examples: Optional[int] = None) -> Tuple[DataFrame, DataFrame, Series, Series]:
    test_size = int(len(y) * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    if (train_examples is not None) and len(x) > train_examples:
        test_size = len(x) - train_examples
    x_train, x_test, y_train, y_test = do_split(x=x, y=y, test_size=test_size, is_cls=is_cls, fold=fold)
    return x_train, x_test, y_train, y_test

def split_to_val(x: DataFrame, y: Series, is_cls: bool, fold: int = -1, val_ratio: float = VAL_RATIO) -> Tuple[DataFrame, DataFrame, Series, Series]:
    val_size = int(len(y) * val_ratio)
    val_size = min(val_size, MAX_VAL_SIZE)
    x_train, x_val, y_train, y_val = do_split(x=x, y=y, test_size=val_size, is_cls=is_cls, fold=fold)
    return x_train, x_val, y_train, y_val


def do_split(x: DataFrame, y: Series, test_size: int, is_cls: bool, fold: int) -> Tuple[DataFrame, DataFrame, Series, Series]:
    if not x.index.equals(y.index):
        raise ValueError("X and y must share identical indices. Construct y as: y = pd.Series(y, index=X.index)")
    random_state = SEED + fold
    if not is_cls:
        return train_test_split(x, y, test_size=test_size, random_state=random_state)
    num_classes = y.nunique()
    if num_classes == 2:
        return _split_for_binary_classification(x, y, test_size=test_size, random_state=random_state)
    has_rare_class = y.value_counts().min() <= 1
    if has_rare_class:
        return _split_with_rare_classes(x, y, test_size=test_size, random_state=random_state)
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def _split_with_rare_classes(x: DataFrame, y: Series, test_size: int, random_state: int) -> Tuple[DataFrame, DataFrame, Series, Series]:
    # TODO: add tests here, seems like a complex function
    singleton_classes = y.value_counts()[y.value_counts() == 1].index
    is_singleton = y.isin(singleton_classes)
    x_single = x[is_singleton]
    y_single = y[is_singleton]
    x_rest = x[~is_singleton]
    y_rest = y[~is_singleton]

    rest_classes = len(set(y_rest))
    test_size = max(test_size, rest_classes)
    x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=test_size, random_state=random_state,
                                                        stratify=y_rest)
    x_train, y_train = merge_splits(x1=x_train, x2=x_single, y1=y_train, y2=y_single, random_state=random_state)
    return x_train, x_test, y_train, y_test


def _split_for_binary_classification(x: DataFrame, y: Series, test_size: int, random_state: int) -> Tuple[DataFrame, DataFrame, Series, Series]:
    test_indices = []
    y_values = sorted(set(y))
    assert len(y_values) == 2, f"Expected 2 classes, got {y_values}"
    random_number_generator = np.random.RandomState(random_state)
    for v in y_values:
        v_indices = y[y == v].index.tolist()
        v_idx = random_number_generator.choice(v_indices)
        test_indices.append(v_idx)
    x_single_test = x.loc[test_indices]
    y_single_test = y.loc[test_indices]
    x_rest = x.drop(index=test_indices)
    y_rest = y.drop(index=test_indices)
    x_train, x_test, y_train, y_test = train_test_split(x_rest, y_rest, test_size=test_size, random_state=random_state,
                                                        stratify=y_rest)
    x_test, y_test = merge_splits(x1=x_test, x2=x_single_test, y1=y_test, y2=y_single_test, random_state=random_state)
    return x_train, x_test, y_train, y_test



def merge_splits(x1: DataFrame, x2: DataFrame, y1: Series, y2: Series, random_state: int) -> Tuple[DataFrame, Series]:
    x_train = pd.concat([x1, x2])
    y_train = pd.concat([y1, y2])
    shuffled = x_train.sample(frac=1, random_state=random_state).index
    x = x_train.loc[shuffled]
    y = y_train.loc[shuffled]
    return x, y