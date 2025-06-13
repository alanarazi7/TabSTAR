import os
from time import sleep

import kagglehub
import openml
from openml.exceptions import OpenMLServerException
from pandas import read_csv, DataFrame

from tabstar.datasets.all_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID, TabularDatasetID
from tabstar_paper.datasets.curation import curate_dataset, TabularDataset


def download_dataset(dataset_id: TabularDatasetID) -> TabularDataset:
    if isinstance(dataset_id, OpenMLDatasetID):
        return load_openml_dataset(dataset_id)
    elif isinstance(dataset_id, KaggleDatasetID):
        return load_kaggle_dataset(dataset_id)
    elif isinstance(dataset_id, UrlDatasetID):
        return load_url_dataset(dataset_id)
    else:
        raise ValueError(f"Unsupported dataset ID type: {type(dataset_id)}")


def load_openml_dataset(dataset_id: OpenMLDatasetID) -> TabularDataset:
    for i in range(10):
        try:
            print(f"💾 Downloading OpenML dataset {dataset_id.name}")
            openml_dataset = openml.datasets.get_dataset(dataset_id.value, download_data=True, download_features_meta_data=True)
            x, y, _, _ = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
            dataset = curate_dataset(x=x, y=y, dataset_id=dataset_id)
            return dataset
        except (OpenMLServerException, ConnectionError, FileNotFoundError) as e:
            print(f"⚠️⚠️⚠️ OpenML exception: {e}")
            sleep(120)
    raise ValueError("Failed to load OpenML dataset")


def load_kaggle_dataset(dataset_id: KaggleDatasetID) -> TabularDataset:
    assert dataset_id.value.count('/') == 2
    dataset_name, file = dataset_id.value.rsplit('/', 1)
    print(f"💾 Downloading Kaggle dataset {dataset_id.name}")
    dir_path = kagglehub.dataset_download(dataset_name)
    file_path = os.path.join(dir_path, file)
    df = _read_csv(file_path, dataset_id=dataset_id)
    dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id)
    return dataset


def load_url_dataset(dataset_id: UrlDatasetID) -> TabularDataset:
    print(f"💾 Downloading URL dataset {dataset_id.name}")
    df = _read_csv(path=str(dataset_id.value), dataset_id=dataset_id)
    dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id)
    return dataset



def _read_csv(path: str, dataset_id: TabularDatasetID) -> DataFrame:
    sep = ","
    if dataset_id in {KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
                      UrlDatasetID.REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT,
                      UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER}:
        sep = ";"
    return read_csv(path, sep=sep)