import os
from typing import Tuple

import kagglehub
import pandas as pd
from pandas import read_csv, DataFrame

from tabular.datasets.manual_curation_mapping import get_curated
from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.raw_loader import get_dataset_description, create_raw_dataset, get_dataframe_types, \
    set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import KaggleDatasetID, get_sid
from tabular.preprocessing.feature_type import get_feature_types


def load_kaggle_dataset(dataset_id: KaggleDatasetID) -> RawDataset:
    sid = get_sid(dataset_id)
    assert dataset_id.value.count('/') == 2
    source_name = dataset_id.value.split('/')[1]
    x = load_from_kaggle(dataset_id)
    curation = get_curated(dataset_id)
    description = _get_kaggle_description(dataset_id=dataset_id, curated=curation, x=x)
    x, y, task_type, curation = set_target_drop_redundant_downsample_too_big(x=x, y=None, curation=curation, sid=sid)
    kaggle_types = get_dataframe_types(x)
    feature_types = get_feature_types(x=x, curation=curation, feat_types=kaggle_types)
    raw = create_raw_dataset(x=x, y=y, curation=curation, desc=description, feat_types=feature_types, sid=sid,
                             task_type=task_type, source_name=source_name)
    return raw


def _get_kaggle_description(dataset_id: KaggleDatasetID, curated: CuratedDataset, x: DataFrame) -> str:
    url = f"https://www.kaggle.com/{dataset_id.value}"
    description = curated.description or ""
    description = get_dataset_description(name=dataset_id.value, url=url, desc=description, x=x)
    return description

def load_from_kaggle(dataset_id: KaggleDatasetID) -> DataFrame:
    dataset, file = _split_to_name_and_file(dataset_id)
    dir_path = kagglehub.dataset_download(dataset)
    file_path = os.path.join(dir_path, file)
    if dataset_id in {KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY}:
        return pd.read_csv(file_path, sep=";")
    df = read_csv(file_path)
    return df

def _get_dataset_name(dataset_id: KaggleDatasetID) -> str:
    dataset, file = _split_to_name_and_file(dataset_id)
    return dataset

def _split_to_name_and_file(dataset_id: KaggleDatasetID) -> Tuple[str, str]:
    dataset, file = dataset_id.value.rsplit('/', 1)
    return dataset, file
