import os
from dataclasses import dataclass, fields
from functools import lru_cache
from os.path import join
from typing import Self

import h5py
from torch.utils.data import Dataset

from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData
from tabstarv1p1_pretrain.utils.tabular import task_type_from_d_output
from tabstarv1p1_pretrain.datasets.objects import SupervisedTask
from tabstarv1p1_pretrain.utils.io_handlers import load_json, dump_json


@lru_cache(maxsize=200)
def _open_h5_file(path: str) -> h5py.File:
    # A pretraining pool can hold far more files than a process's file-descriptor ulimit
    # allows to stay open at once, so the cache evicts the least-recently-used path; h5py
    # closes an evicted File as soon as this was its last reference.
    return h5py.File(path, 'r')


@dataclass
class RealDatasetProperties:
    name: str
    d_output: int
    train_size: int

    @classmethod
    def from_json(cls, data_dir: str) -> Self:
        properties_path = join(data_dir, RealDataset.PROPERTIES)
        d = load_json(properties_path)
        allowed_fields = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in allowed_fields}
        return cls(**d)

    @property
    def is_cls(self) -> bool:
        return self.d_output > 1

    @property
    def task_type(self) -> SupervisedTask:
        return task_type_from_d_output(self.d_output)


class RealDataset(Dataset):

    TRAIN = "train"
    X_TXT_KEY = "X_txt"
    X_NUM_KEY = "X_num"
    Y_KEY = "y"
    H5_FILE = "data.h5"
    PROPERTIES = "properties.json"

    def __init__(self, data_dir: str):
        self.file_path = join(data_dir, self.TRAIN, self.H5_FILE)
        self.properties: RealDatasetProperties = RealDatasetProperties.from_json(data_dir)
        self.size: int = self.properties.train_size

    @property
    def is_cls(self) -> bool:
        return self.properties.is_cls

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        h5_file = _open_h5_file(self.file_path)
        x_txt = h5_file[self.X_TXT_KEY][idx]
        x_num = h5_file[self.X_NUM_KEY][idx]
        y = h5_file[self.Y_KEY][idx]
        return x_num, x_txt, y, self.properties


def save_pretrain_dataset(data_dir: str, train_data: TabSTARData, properties: RealDatasetProperties):
    property_path = join(data_dir, RealDataset.PROPERTIES)
    dump_json(vars(properties), path=property_path)
    split_dir = join(data_dir, RealDataset.TRAIN)
    os.makedirs(split_dir, exist_ok=True)
    h5_file_path = join(split_dir, RealDataset.H5_FILE)
    key2data = {
        RealDataset.X_TXT_KEY: train_data.x_txt,
        RealDataset.X_NUM_KEY: train_data.x_num,
        RealDataset.Y_KEY: train_data.y
    }
    with h5py.File(h5_file_path, 'w') as h5f:
        for key, value in key2data.items():
            h5f.create_dataset(name=key, data=value)