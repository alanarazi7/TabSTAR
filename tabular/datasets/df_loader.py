import pandas as pd
from pandas import DataFrame

from tabular.datasets.manual_curation_mapping import get_curated
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.raw_loader import get_dataset_description, create_raw_dataset, get_dataframe_types, \
    set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import get_sid, UrlDatasetID
from tabular.preprocessing.feature_type import get_feature_types


def load_df_dataset(dataset_id: UrlDatasetID) -> RawDataset:
    sid = get_sid(dataset_id)
    x = load_file_from_url(dataset_id)
    curation = get_curated(dataset_id)
    description = get_dataset_description(name=dataset_id.name, url=dataset_id.value, desc=curation.description, x=x)
    x, y, task_type, curation = set_target_drop_redundant_downsample_too_big(x=x, y=None, curation=curation, sid=sid)
    kaggle_types = get_dataframe_types(x)
    feature_types = get_feature_types(x=x, curation=curation, feat_types=kaggle_types)
    raw = create_raw_dataset(x=x, y=y, curation=curation, desc=description, feat_types=feature_types, sid=sid,
                             task_type=task_type)
    return raw


def load_file_from_url(dataset_id: UrlDatasetID) -> DataFrame:
    url = str(dataset_id.value)
    special_urls = ["scimagojr.com/journalrank", "opendata.vancouver.ca/api/records"]
    for special in special_urls:
        if special in url:
            return pd.read_csv(url, sep=";")
    return pd.read_csv(url)
