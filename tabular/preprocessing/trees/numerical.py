from dataclasses import dataclass
from typing import List, Tuple, Optional

from pandas import Series

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.splits import DataSplit, get_x_train
from tabular.utils.utils import verbose_print


@dataclass
class MedianFiller:
    src: Series
    target: Series
    median: float


def fill_median_for_numerical_nulls(raw: RawDataset, splits: List[DataSplit]):
    x_train = get_x_train(x=raw.x, splits=splits)
    verbose_print(f"❓ Filling numerical nulls with median for {raw.sid}. Columns: {raw.numerical}")
    for col in raw.numerical:
        median_filler = fill_median(x_train=x_train[col], x_test=raw.x[col])
        raw.x[col] = median_filler.target


def fill_median(x_train: Series, x_test: Optional[Series] = None) -> MedianFiller:
    """Fill the test set with the median of the train set."""
    train_median = x_train.median()
    x_train = x_train.copy().fillna(train_median)
    if x_test is not None:
        x_test = x_test.copy().fillna(train_median)
    return MedianFiller(src=x_train, target=x_test, median=train_median)
