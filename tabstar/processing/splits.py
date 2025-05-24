from typing import Tuple

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabstar.constants import SEED

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000
NN_DEV_RATIO = 0.1
MAX_DEV_SIZE = 1000


def split_to_test(x: DataFrame, y: Series) -> Tuple[DataFrame, DataFrame, Series, Series]:
    n = len(y)
    test_size = int(n * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=SEED, stratify=y)
    return x_train, x_test, y_train, y_test


# def create_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> List[DataSplit]:
#     n = len(raw.y)
#     indices = list(range(n))
#     is_pretrain = bool(train_examples < 0)
#     use_dev = _uses_dev(processing)
#     if is_pretrain:
#         test = []
#     else:
#         indices, test = _get_test(raw=raw, indices=indices, n=n, run_num=run_num)
#         indices, exclude = _get_exclude(raw=raw, indices=indices, run_num=run_num, train_examples=train_examples)
#     train, dev = _get_train_dev(raw=raw, indices=indices, use_dev=use_dev, run_num=run_num, is_pretrain=is_pretrain)
#     splits = {DataSplit.TRAIN: train, DataSplit.DEV: dev, DataSplit.TEST: test}
#     split_array = _sample_xy_and_get_array(raw=raw, n=n, splits=splits)
#     return split_array
#
#
# def _get_train_dev(raw: RawDataset, indices: List[int], use_dev: bool, run_num: int,
#                    is_pretrain: bool) -> Tuple[List[int], List[int]]:
#     if not use_dev:
#         return indices, []
#     dev_size = int(len(indices) * NN_DEV_RATIO)
#     dev_size = min(dev_size, MAX_DEV_SIZE)
#     return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=dev_size)
#
#
# def _do_split(raw: RawDataset, indices: List[int], run_num: int, test_size: int) -> Tuple[List[int], List[int]]:
#     random_state = SEED + run_num
#     stratify = raw.y.iloc[indices] if raw.task_type != SupervisedTask.REGRESSION else None
#     try:
#         train, test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify)
#     except ValueError as e:
#         assert raw.task_type != SupervisedTask.REGRESSION
#         train, test = train_test_split(indices, test_size=test_size, random_state=random_state)
#         train_classes = set(raw.y.iloc[train])
#         missing_class_indices = [idx for idx in test if raw.y.iloc[idx] not in train_classes]
#         if missing_class_indices:
#             train.extend(missing_class_indices)
#             test = [idx for idx in test if idx not in missing_class_indices]
#     return train, test
#