from time import sleep
from typing import Tuple, Optional

import openml
from openml import OpenMLDataset
from openml.exceptions import OpenMLServerException
from pandas import DataFrame, Series

from tabstar_paper.datasets.curation_mapping import get_curated
from tabular.datasets.raw_loader import create_raw_dataset, set_target_drop_redundant_downsample_too_big
from tabular.datasets.tabular_datasets import OpenMLDatasetID, get_sid
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.feature_type import get_feature_types
from tabular.preprocessing.objects import FeatureType
from tabular.utils.utils import verbose_print


def load_openml_dataset(dataset_id: OpenMLDatasetID) -> RawDataset:
    sid = get_sid(dataset_id)
    dataset, x, y = load_from_openml(dataset_id)
    openml_types = {feat.name: FeatureType(feat.data_type) for feat in dataset.features.values()}
    curation = get_curated(dataset_id)
    x, y, task_type, curation = set_target_drop_redundant_downsample_too_big(x=x, y=y, curation=curation, sid=sid)
    feature_types = get_feature_types(x=x, curation=curation, feat_types=openml_types)
    raw = create_raw_dataset(x=x, y=y, curation=curation, feat_types=feature_types, sid=sid, task_type=task_type)
    return raw

def load_from_openml(dataset_id: OpenMLDatasetID) -> Tuple[OpenMLDataset, DataFrame, Optional[Series]]:
    for i in range(50):
        try:
            verbose_print(f"💾 Downloading OpenML dataset {dataset_id.name}")
            dataset = openml.datasets.get_dataset(dataset_id.value, download_data=True, download_features_meta_data=True)
            x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            verbose_print(f"✅ Loaded {len(x)} examples with {len(x.columns)} features")
            return dataset, x, y
        except (OpenMLServerException, ConnectionError, FileNotFoundError) as e:
            print(f"⚠️⚠️⚠️ OpenML exception: {e}")
            sleep(120)
    raise ValueError(f"Failed to load OpenML dataset {dataset_id} after 20 attempts.")
