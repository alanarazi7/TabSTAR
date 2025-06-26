from typing import Tuple

from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.preprocessing.splits import TEST_RATIO, MAX_TEST_SIZE, do_split
from tabstar_paper.utils.logging import log_calls


@log_calls
def subsample_dataset(x: DataFrame, y: Series, is_cls: bool, train_examples: int,
                      seed: int = SEED) -> Tuple[DataFrame, Series]:
    test_examples = int(len(y) * TEST_RATIO)
    test_examples = min(test_examples, MAX_TEST_SIZE)
    all_examples = train_examples + test_examples
    if len(y) <= all_examples:
        return x, y
    test_size = len(y) - all_examples
    x_sample, x_exclude, y_sample, y_exclude = do_split(x=x, y=y, test_size=test_size, is_cls=is_cls, seed=seed)
    return x_sample, y_sample
