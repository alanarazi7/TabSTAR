from collections import Counter
from copy import deepcopy
from typing import Optional, List, Tuple

import numpy as np
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tabular.datasets.manual_curation_obj import CuratedDataset, CuratedTarget
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.curation import assert_no_wrong_key
from tabular.preprocessing.nulls import convert_series_to_numeric
from tabular.preprocessing.objects import SupervisedTask, PreprocessingMethod, CV_METHODS
from tabular.preprocessing.splits import DataSplit, get_y_train
from tabular.utils.processing import transform_encoded
from tabular.utils.utils import cprint, verbose_print

MIN_MULTICLASS_FREQUENCY = 5
Z_MAX_ABS_VAL = 3


def handle_raw_target(x: DataFrame, y: Optional[Series], curation: CuratedDataset,
                      sid: str) -> Tuple[DataFrame, Series, SupervisedTask]:
    if y is None:
        x, y = _set_target_if_missing(x=x, curation=curation)
    if curation.target.processing_func:
        y = y.apply(curation.target.processing_func)
    if curation.target.numeric_missing:
        assert curation.target.task_type == SupervisedTask.REGRESSION
        y = convert_series_to_numeric(y, missing_value=curation.target.numeric_missing)
    x, y = _remove_missing_target_rows(x, y)
    task_type = curation.target.task_type
    y = _curate_target_values(y=y, target=curation.target, task_type=task_type)
    if task_type == SupervisedTask.MULTICLASS:
        x, y = _remove_rare_target_rows(x=x, y=y, sid=sid)
    assert _get_sid_task_type(sid) == task_type
    return x, y, task_type


def process_y(raw: RawDataset, splits: List[DataSplit], processing: PreprocessingMethod) -> Optional[List[str]]:
    if raw.task_type != SupervisedTask.REGRESSION:
        return _process_cls_y(raw=raw)
    elif processing in CV_METHODS:
        verbose_print(f"Avoid target handling for {raw.sid}, CV method {processing}. Values: {list(set(raw.y))[:10]}")
        return None
    else:
        return _standardize_label(raw=raw, splits=splits)


def get_label_repr(y: Series, task_type: SupervisedTask) -> str:
    if task_type == SupervisedTask.REGRESSION:
        avg = y.mean()
        return f"{avg:.2f}"
    value_proportions = y.value_counts(normalize=True)
    most_common = value_proportions.head(10).apply(lambda v: f"{v:.2%}").to_list()
    if task_type == SupervisedTask.BINARY:
        return most_common[0]
    else:
        return f"{len(value_proportions)}C: {most_common}"


def _process_cls_y(raw: RawDataset) -> List[str]:
    label_encoder = LabelEncoder()
    all_classes = set(raw.y)
    label_encoder.fit(list(all_classes))
    raw.y = transform_target(y=raw.y, transformer=label_encoder)
    return [str(v) for v in label_encoder.classes_]


def _standardize_label(raw: RawDataset, splits: List[DataSplit]):
    y_train = get_y_train(y=raw.y, splits=splits)
    scaler = fit_standard_scaler(y_train=y_train)
    raw.y = transform_target(y=raw.y, transformer=scaler)

def standardize_y_train_test(y_train: Series, y_test: Series) -> Tuple[Series, Series]:
    y_train = deepcopy(y_train)
    y_test = deepcopy(y_test)
    scaler = fit_standard_scaler(y_train=y_train)
    y_train = transform_target(y=y_train, transformer=scaler)
    y_test = transform_target(y=y_test, transformer=scaler)
    return y_train, y_test

def fit_standard_scaler(y_train: Series) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y_train.values.reshape(-1, 1))
    return scaler

def transform_target(y: Series, transformer: LabelEncoder | StandardScaler) -> Series:
    y = deepcopy(y)
    if isinstance(transformer, LabelEncoder):
        y_val = transformer.transform(y)
        is_float = False
    elif isinstance(transformer, StandardScaler):
        y_val = transformer.transform(y.values.reshape(-1, 1)).flatten()
        y_val = np.clip(y_val, -Z_MAX_ABS_VAL, Z_MAX_ABS_VAL)
        is_float = True
    else:
        raise NotImplementedError(f"Unsupported transformer: {transformer}")
    return transform_encoded(col=y, values=y_val, is_float=is_float)


def _set_target_if_missing(x: DataFrame, curation: CuratedDataset) -> Tuple[DataFrame, Series]:
    target_var = curation.target.raw_name
    if target_var not in x.columns:
        raise ValueError(f"Target variable {target_var} not found! Have: {x.columns}")
    y = x[target_var].copy()
    x = x.drop(columns=[target_var])
    return x, y


def _remove_missing_target_rows(x: DataFrame, y: Series):
    missing = y.isnull()
    x = x[~missing]
    y = y[~missing]
    return x, y

def _remove_rare_target_rows(x: DataFrame, y: Series, sid: str):
    label_cnt = Counter(y)
    invalid_labels = [k for k, v in label_cnt.items() if v < MIN_MULTICLASS_FREQUENCY]
    if not invalid_labels:
        return x, y
    cprint(f"🦄 Removing for {sid} the rare labels: {invalid_labels}: {label_cnt}")
    valid_labels = [k for k in label_cnt if k not in invalid_labels]
    if len(valid_labels) <= 2:
        raise ValueError(f"Too few valid labels: {valid_labels}, with count {label_cnt}")
    mask = y.isin(valid_labels)
    x = x[mask]
    y = y[mask]
    return x, y


def _curate_target_values(y: Series, target: CuratedTarget, task_type: SupervisedTask) -> Series:
    assert not y.isna().any(), "Missing values in target are not allowed!"
    if task_type == SupervisedTask.REGRESSION:
        return y.astype(float)
    assert task_type in {SupervisedTask.BINARY, SupervisedTask.MULTICLASS}
    y = y.astype(str)
    assert_no_wrong_key(s=y, mapper=target.label_mapping)
    y = y.apply(lambda v: target.label_mapping.get(str(v), str(v)))
    return y


def _get_sid_task_type(sid: str) -> SupervisedTask:
    d = {'BIN': SupervisedTask.BINARY, 'MUL': SupervisedTask.MULTICLASS, 'REG': SupervisedTask.REGRESSION}
    for prefix, task_type in d.items():
        if sid.startswith(f"{prefix}_"):
            return task_type
    sid = sid.split('_')[1]
    return d[sid]
