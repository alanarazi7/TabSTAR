import os
from time import sleep

import kagglehub
import openml
from openml.exceptions import OpenMLServerException
from pandas import read_csv, DataFrame

from tabstar.datasets.all_datasets import OpenMLDatasetID, KaggleDatasetID, UrlDatasetID, TabularDatasetID
from tabstar_paper.datasets.curation import curate_dataset, TabularDataset
from tabstar_paper.utils import log_calls

OPENML_VALUES = {item.value for item in OpenMLDatasetID}
KAGGLE_VALUES = {item.value for item in KaggleDatasetID}
URL_VALUES = {item.value for item in UrlDatasetID}


@log_calls
def download_dataset(dataset_id: TabularDatasetID) -> TabularDataset:
    # TODO: allow the option to downsample the number of examples of the dataset
    if dataset_id.value in OPENML_VALUES:
        return load_openml_dataset(dataset_id)
    elif dataset_id.value in KAGGLE_VALUES:
        return load_kaggle_dataset(dataset_id)
    elif dataset_id.value in URL_VALUES:
        return load_url_dataset(dataset_id)
    else:
        raise ValueError(f"Unsupported dataset ID type: {type(dataset_id)}")


@log_calls
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


@log_calls
def load_kaggle_dataset(dataset_id: KaggleDatasetID) -> TabularDataset:
    assert dataset_id.value.count('/') == 2
    dataset_name, file = dataset_id.value.rsplit('/', 1)
    print(f"💾 Downloading Kaggle dataset {dataset_id.name}")
    dir_path = kagglehub.dataset_download(dataset_name)
    file_path = os.path.join(dir_path, file)
    df = _read_csv(file_path, dataset_id=dataset_id)
    dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id)
    return dataset


@log_calls
def load_url_dataset(dataset_id: UrlDatasetID) -> TabularDataset:
    print(f"💾 Downloading URL dataset {dataset_id.name}")
    df = _read_csv(path=str(dataset_id.value), dataset_id=dataset_id)
    dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id)
    return dataset


@log_calls
def _read_csv(path: str, dataset_id: TabularDatasetID) -> DataFrame:
    sep = ","
    if dataset_id in {KaggleDatasetID.REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
                      UrlDatasetID.REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT,
                      UrlDatasetID.REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER}:
        sep = ";"
    return read_csv(path, sep=sep)


@log_calls
def get_dataset_from_arg(arg: str | int) -> TabularDatasetID:
    if isinstance(arg, str) and arg.isdigit():
        arg = int(arg)
    if arg is None:
        raise ValueError("Dataset ID cannot be None.")
    for dataset_type in [OpenMLDatasetID, KaggleDatasetID, UrlDatasetID]:
        for dataset in dataset_type:
            if dataset.value == arg or dataset.name == arg:
                return dataset
    raise ValueError(f"Dataset ID {arg} not found in any known datasets.")