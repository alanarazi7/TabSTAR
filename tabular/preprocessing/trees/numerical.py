from dataclasses import dataclass
from typing import Optional

from pandas import Series

@dataclass
class MedianFiller:
    src: Series
    target: Series
    median: float


def fill_median(x_train: Series, x_test: Optional[Series] = None) -> MedianFiller:
    """Fill the test set with the median of the train set."""
    train_median = x_train.median()
    x_train = x_train.copy().fillna(train_median)
    if x_test is not None:
        x_test = x_test.copy().fillna(train_median)
    return MedianFiller(src=x_train, target=x_test, median=train_median)
