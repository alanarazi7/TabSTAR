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
    x_train_filled = x_train.copy().fillna(train_median)
    if x_test is not None:
        x_test_filled = x_test.copy().fillna(train_median)
    else:
        x_test_filled = None
    return MedianFiller(src=x_train_filled, target=x_test_filled, median=train_median)
