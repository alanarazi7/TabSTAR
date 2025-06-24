import os.path
from dataclasses import dataclass
from os.path import join
from typing import Optional, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tabstar.tabstar_verbalizer import TabSTARData
from tabstar_paper.utils import dump_json


@dataclass
class DatasetProperties:
    name: str
    d_output: int
    idx2text: Dict[int, str]
    train_size: int
    val_size: int


class HDF5Dataset(Dataset):

    X_TXT_KEY = "X_txt"
    X_NUM_KEY = "X_num"
    Y_KEY = "y"
    H5_FILE = "data.h5"

    def __init__(self, split_dir: str):
        self.file_path = join(split_dir, self.H5_FILE)
        # self.properties: PretrainDatasetProperties = get_properties(data_dir=os.path.dirname(split_dir))
        # self.size: int = self.properties.split_sizes[os.path.basename(split_dir)]
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


def save_pretrain_dataset(data_dir: str, train_data: TabSTARData, val_data: TabSTARData, properties: DatasetProperties):
    raise NotImplementedError('save_data_splits, save_properties')


# def create_dataset(data_dir: str, dataset: TabularDatasetID, processing: PreprocessingMethod, run_num: int,
#                    train_examples: int, device: torch.device, number_verbalization: Optional[NumberVerbalization] = None):
#     fix_seed()
#     raw_dataset = get_raw_dataset(dataset)
#     dataset = TabularDataset.from_raw(raw=raw_dataset, processing=processing, run_num=run_num,
#                                       train_examples=train_examples, device=device,
#                                       number_verbalization=number_verbalization)
#     verbose_print(f"Saving dataset {dataset.properties.sid} to {data_dir}")
#     save_data_splits(dataset=dataset, data_dir=data_dir, processing=processing)
#     save_properties(data_dir=data_dir, dataset=dataset)
#     verbose_print(f"🎉 Saved!")
#
#
# def save_properties(data_dir: str, dataset: TabularDataset):
#     create_dir(data_dir)
#     dump_json(dataset.properties.to_dict(), path=properties_path(data_dir))
#
# def get_split_dir(data_dir: str, split: DataSplit) -> str:
#     split_dir = join(data_dir, split)
#     create_dir(split_dir)
#     return split_dir
#
# def get_properties(data_dir: str) -> DatasetProperties:
#     return DatasetProperties.from_json(properties_path(data_dir))
#
#
# def save_data_splits(dataset: TabularDataset, data_dir: str, processing: PreprocessingMethod):
#     for split in DataSplit:
#         split_dir = get_split_dir(data_dir, split)
#         indices = [i for i, s in enumerate(dataset.splits) if s == split]
#         x = pd_indices_to_array(dataset.x, indices)
#         y = pd_indices_to_array(dataset.y, indices)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x_num = dataset.x_num[indices]
#         save_for_tabstar(split_dir, x=x, y=y, x_num=x_num)
#
#
# def save_for_tabstar(split_dir: str, x: np.ndarray, y: np.ndarray, x_num: np.ndarray):
#     h5_file_path = join(split_dir, HDF5Dataset.H5_FILE)
#     key2data = {HDF5Dataset.X_TXT_KEY: x, HDF5Dataset.X_NUM_KEY: x_num, HDF5Dataset.Y_KEY: y}
#     with h5py.File(h5_file_path, 'w') as h5f:
#         for key, data in key2data.items():
#             h5f.create_dataset(name=key, data=data)

