import os

import kagglehub
import pandas as pd
from pandas import read_csv, DataFrame

from tabstar_paper.datasets.curation_mapping import get_curated
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.raw_loader import create_raw_dataset, get_dataframe_types, set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import KaggleDatasetID, get_sid
from tabular.preprocessing.feature_type import get_feature_types


def load_kaggle_dataset(dataset_id: KaggleDatasetID) -> RawDataset:
    sid = get_sid(dataset_id)
    assert dataset_id.value.count('/') == 2
    x = load_from_kaggle(dataset_id)
    curation = get_curated(dataset_id)
    x, y, task_type, curation = set_target_drop_redundant_downsample_too_big(x=x, y=None, curation=curation, sid=sid)
    kaggle_types = get_dataframe_types(x)
    feature_types = get_feature_types(x=x, curation=curation, feat_types=kaggle_types)
    raw = create_raw_dataset(x=x, y=y, curation=curation, feat_types=feature_types, sid=sid, task_type=task_type)
    return raw


def load_from_kaggle(dataset_id: KaggleDatasetID) -> DataFrame:
    dataset, file = dataset_id.value.rsplit('/', 1)
    dir_path = kagglehub.dataset_download(dataset)
    file_path = os.path.join(dir_path, file)
    if dataset_id in {KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY}:
        return pd.read_csv(file_path, sep=";")
    df = read_csv(file_path)
    return df
