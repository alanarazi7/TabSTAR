from collections import Counter
from enum import StrEnum
from typing import List, Dict, Tuple

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.objects import SupervisedTask, PreprocessingMethod, CV_METHODS
from tabular.utils.utils import SEED, verbose_print

TEST_RATIO = 0.1
MAX_TEST_SIZE = 2000

NN_DEV_RATIO = 0.1
NN_PRETRAIN_DEV_RATIO = 0.05
MAX_DEV_SIZE = 1000

MIN_TOTAL_EXAMPLES = 100


class DataSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


def create_splits(raw: RawDataset, run_num: int, train_examples: int, processing: PreprocessingMethod) -> List[DataSplit]:
    n = len(raw.y)
    if n < MIN_TOTAL_EXAMPLES:
        raise ValueError(f"Dataset {raw.sid} has too few examples: {n}")
    indices = list(range(n))
    is_pretrain = bool(train_examples < 0)
    use_dev = _uses_dev(processing)
    if is_pretrain:
        test = []
    else:
        indices, test = _get_test(raw=raw, indices=indices, n=n, run_num=run_num)
        indices, exclude = _get_exclude(raw=raw, indices=indices, run_num=run_num, train_examples=train_examples)
    train, dev = _get_train_dev(raw=raw, indices=indices, use_dev=use_dev, run_num=run_num, is_pretrain=is_pretrain)
    splits = {DataSplit.TRAIN: train, DataSplit.DEV: dev, DataSplit.TEST: test}
    split_array = _sample_xy_and_get_array(raw=raw, n=n, splits=splits)
    verbose_print(f"Created splits for {raw.sid} of length {n} and {train_examples=}: {Counter(split_array)}")
    return split_array

def _get_test(raw: RawDataset, indices: List[int], n: int, run_num: int) -> Tuple[List[int], List[int]]:
    test_size = int(n * TEST_RATIO)
    test_size = min(test_size, MAX_TEST_SIZE)
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=test_size)

def _get_exclude(raw: RawDataset, indices: List[int], run_num: int, train_examples: int) -> Tuple[List[int], List[int]]:
    if len(indices) < train_examples:
        return indices, []
    exclude_examples = len(indices) - train_examples
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=exclude_examples)


def _get_train_dev(raw: RawDataset, indices: List[int], use_dev: bool, run_num: int,
                   is_pretrain: bool) -> Tuple[List[int], List[int]]:
    if not use_dev:
        return indices, []
    nn_dev_ratio = NN_PRETRAIN_DEV_RATIO if is_pretrain else NN_DEV_RATIO
    dev_size = int(len(indices) * nn_dev_ratio)
    dev_size = min(dev_size, MAX_DEV_SIZE)
    return _do_split(raw=raw, indices=indices, run_num=run_num, test_size=dev_size)


def get_x_train(x: DataFrame, splits: List[DataSplit]) -> DataFrame:
    return get_x_split(x=x, splits=splits, split=DataSplit.TRAIN)

def get_x_split(x: DataFrame, splits: List[DataSplit], split: DataSplit) -> DataFrame:
    indices = [i for i, s in enumerate(splits) if s == split]
    return x.iloc[indices].reset_index(drop=True)

def get_y_train(y: Series, splits: List[DataSplit]) -> Series:
    return get_y_split(y=y, splits=splits, split=DataSplit.TRAIN)

def get_y_split(y: Series, splits: List[DataSplit], split: DataSplit) -> Series:
    indices = [i for i, s in enumerate(splits) if s == split]
    return y.iloc[indices].reset_index(drop=True)


def _do_split(raw: RawDataset, indices: List[int], run_num: int, test_size: int) -> Tuple[List[int], List[int]]:
    random_state = SEED + run_num
    stratify = raw.y.iloc[indices] if raw.task_type != SupervisedTask.REGRESSION else None
    try:
        train, test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError as e:
        assert raw.task_type != SupervisedTask.REGRESSION
        train, test = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_classes = set(raw.y.iloc[train])
        missing_class_indices = [idx for idx in test if raw.y.iloc[idx] not in train_classes]
        if missing_class_indices:
            train.extend(missing_class_indices)
            test = [idx for idx in test if idx not in missing_class_indices]
    return train, test

def _sample_xy_and_get_array(raw: RawDataset, n: int, splits: Dict[DataSplit, List[int]]) -> List[DataSplit]:
    idx2split = {i: split for split, indices in splits.items() for i in indices}
    splits_array = [idx2split.get(i) for i in range(n)]
    valid_mask = [v is not None for v in splits_array]
    raw.x = raw.x[valid_mask].reset_index(drop=True)
    raw.y = raw.y[valid_mask].reset_index(drop=True)
    split_array = [s for s, valid in zip(splits_array, valid_mask) if valid]
    return split_array

def _uses_dev(processing: PreprocessingMethod) -> bool:
    if processing in CV_METHODS:
        return False
    process2dev = {PreprocessingMethod.TABSTAR: True,
                   PreprocessingMethod.CATBOOST: True,
                   PreprocessingMethod.TREES: True,
                   # TabPFN-v2 and CARTE don't use dev
                   PreprocessingMethod.TABPFNV2: False,
                   PreprocessingMethod.CARTE: False,
                   }
    return process2dev[processing]