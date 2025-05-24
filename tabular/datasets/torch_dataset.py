import os.path
from os.path import join
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from torch.utils.data import Dataset

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
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.utils.io_handlers import dump_json
from tabular.utils.paths import create_dir, dataset_run_properties_dir, properties_path
from tabular.utils.processing import pd_indices_to_array
from tabular.utils.utils import fix_seed, verbose_print


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


class HDF5Dataset(Dataset):

    X_TXT_KEY = "X_txt"
    X_NUM_KEY = "X_num"
    Y_KEY = "y"
    H5_FILE = "data.h5"

    def __init__(self, split_dir: str):
        self.file_path = join(split_dir, self.H5_FILE)
        self.properties: DatasetProperties = get_properties(data_dir=os.path.dirname(split_dir))
        self.size: int = self.properties.split_sizes[os.path.basename(split_dir)]
        self.h5_file: Optional[h5py.File] = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Open the HDF5 file and read a specific single sample
        self.open()
        x_txt = self.h5_file[self.X_TXT_KEY][idx]
        x_txt = np.array([self.properties.idx2text[str(int(i))] for i in x_txt])
        x_num = self.h5_file[self.X_NUM_KEY][idx]
        y = self.h5_file[self.Y_KEY][idx]
        return x_txt, x_num, y, self.properties

    def open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def get_data_dir(dataset: TabularDatasetID, processing: PreprocessingMethod, run_num: int,
                 train_examples: int, device: torch.device, number_verbalization: Optional[NumberVerbalization] = None) -> str:
    sid = get_sid(dataset)
    data_dir = join(dataset_run_properties_dir(run_num=run_num, train_examples=train_examples), processing, sid)
    if number_verbalization is not None and number_verbalization != NumberVerbalization.FULL:
        assert processing == PreprocessingMethod.TABSTAR
        data_dir = join(data_dir, number_verbalization)
    if not os.path.exists(properties_path(data_dir)):
        create_dir(data_dir)
        try:
            create_dataset(data_dir=data_dir, dataset=dataset, processing=processing, run_num=run_num,
                           train_examples=train_examples, device=device, number_verbalization=number_verbalization)
        except Exception as e:
            raise Exception(f"🚨🚨🚨 Error loading dataset {dataset} due to: {e}")
    return data_dir


def create_dataset(data_dir: str, dataset: TabularDatasetID, processing: PreprocessingMethod, run_num: int,
                   train_examples: int, device: torch.device, number_verbalization: Optional[NumberVerbalization] = None):
    fix_seed()
    raw_dataset = get_raw_dataset(dataset)
    dataset = TabularDataset.from_raw(raw=raw_dataset, processing=processing, run_num=run_num,
                                      train_examples=train_examples, device=device,
                                      number_verbalization=number_verbalization)
    if processing == PreprocessingMethod.TABSTAR:
        fill_idx2text(dataset)
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

def fill_idx2text(dataset: TabularDataset):
    all_texts = set(dataset.x.values.ravel())
    dataset.properties.idx2text = {i: t for i, t in enumerate(all_texts)}
    text2idx = {t: i for i, t in dataset.properties.idx2text.items()}
    for col in dataset.x.columns:
        dataset.x[col] = dataset.x[col].apply(lambda x: text2idx[x])


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
        if processing == PreprocessingMethod.TABSTAR:
            x = x.to_numpy()
            y = y.to_numpy()
            x_num = dataset.x_num[indices]
            save_for_tabstar(split_dir, x=x, y=y, x_num=x_num)
        else:
            save_for_baselines(split_dir, x=x, y=y)


def save_for_tabstar(split_dir: str, x: np.ndarray, y: np.ndarray, x_num: np.ndarray):
    h5_file_path = join(split_dir, HDF5Dataset.H5_FILE)
    with h5py.File(h5_file_path, 'w') as h5f:
        for key, data in [(HDF5Dataset.X_TXT_KEY, x), (HDF5Dataset.X_NUM_KEY, x_num), (HDF5Dataset.Y_KEY, y)]:
            h5f.create_dataset(name=key, data=data)

def save_for_baselines(split_dir: str, x: DataFrame, y: Series):
    for filename, arr in [(PandasDataset.X_PATH, x), (PandasDataset.Y_PATH, y)]:
        path = join(split_dir, filename)
        arr.to_json(path, orient='records', lines=True)
