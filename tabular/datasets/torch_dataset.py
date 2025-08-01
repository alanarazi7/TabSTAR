import os.path
from os.path import join

import pandas as pd
import torch
from pandas import DataFrame, Series
from torch.utils.data import Dataset

from tabstar_paper.utils.io_handlers import dump_json, create_dir
from tabular.datasets.data_processing import TabularDataset
from tabular.datasets.df_loader import load_df_dataset
from tabular.datasets.kaggle_loader import load_kaggle_dataset
from tabular.datasets.raw_dataset import RawDataset
from tabular.datasets.tabular_datasets import get_sid, TabularDatasetID, OpenMLDatasetID, KaggleDatasetID, \
    UrlDatasetID
from tabular.datasets.properties import DatasetProperties
from tabular.datasets.openml_loader import load_openml_dataset
from tabular.preprocessing.splits import DataSplit
from tabular.preprocessing.objects import PreprocessingMethod
from tabular.utils.paths import dataset_run_properties_dir, properties_path
from tabular.utils.processing import pd_indices_to_array
from tabular.utils.utils import verbose_print


class PandasDataset(Dataset):
    X_PATH = "X.json"
    Y_PATH = "y.json"

    def __init__(self, split_dir: str):
        self.x = pd.read_json(join(split_dir, self.X_PATH), orient='records', lines=True)
        self.y = pd.read_json(join(split_dir, self.Y_PATH), orient='records', lines=True, typ='series')
        self.properties: DatasetProperties = get_properties(data_dir=os.path.dirname(split_dir))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.properties


def get_data_dir(dataset: TabularDatasetID, processing: PreprocessingMethod, run_num: int,
                 train_examples: int, device: torch.device) -> str:
    sid = get_sid(dataset)
    data_dir = join(dataset_run_properties_dir(run_num=run_num, train_examples=train_examples), processing, sid)
    if not os.path.exists(properties_path(data_dir)):
        create_dir(data_dir)
        try:
            create_dataset(data_dir=data_dir, dataset=dataset, processing=processing, run_num=run_num,
                           train_examples=train_examples, device=device)
        except Exception as e:
            raise Exception(f"🚨🚨🚨 Error loading dataset {dataset} due to: {e}")
    return data_dir


def create_dataset(data_dir: str, dataset: TabularDatasetID, processing: PreprocessingMethod, run_num: int,
                   train_examples: int, device: torch.device):
    raw_dataset = get_raw_dataset(dataset)
    dataset = TabularDataset.from_raw(raw=raw_dataset, processing=processing, run_num=run_num,
                                      train_examples=train_examples, device=device)
    verbose_print(f"Saving dataset {dataset.properties.sid} to {data_dir}")
    save_data_splits(dataset=dataset, data_dir=data_dir, processing=processing)
    save_properties(data_dir=data_dir, dataset=dataset)
    verbose_print(f"🎉 Saved!")


def get_raw_dataset(dataset: TabularDatasetID) -> RawDataset:
    if isinstance(dataset, OpenMLDatasetID):
        return load_openml_dataset(dataset_id=dataset)
    elif isinstance(dataset, KaggleDatasetID):
        return load_kaggle_dataset(dataset_id=dataset)
    elif isinstance(dataset, UrlDatasetID):
        return load_df_dataset(dataset_id=dataset)
    raise TypeError(f"What is this dataset from type {type(dataset)}?")


def save_properties(data_dir: str, dataset: TabularDataset):
    create_dir(data_dir)
    dump_json(dataset.properties.to_dict(), path=properties_path(data_dir))

def get_split_dir(data_dir: str, split: DataSplit) -> str:
    split_dir = join(data_dir, split)
    create_dir(split_dir)
    return split_dir

def get_properties(data_dir: str) -> DatasetProperties:
    return DatasetProperties.from_json(properties_path(data_dir))


def save_data_splits(dataset: TabularDataset, data_dir: str, processing: PreprocessingMethod):
    for split in DataSplit:
        split_dir = get_split_dir(data_dir, split)
        indices = [i for i, s in enumerate(dataset.splits) if s == split]
        x = pd_indices_to_array(dataset.x, indices)
        y = pd_indices_to_array(dataset.y, indices)
        save_for_baselines(split_dir, x=x, y=y)


def save_for_baselines(split_dir: str, x: DataFrame, y: Series):
    for filename, arr in [(PandasDataset.X_PATH, x), (PandasDataset.Y_PATH, y)]:
        path = join(split_dir, filename)
        arr.to_json(path, orient='records', lines=True)
