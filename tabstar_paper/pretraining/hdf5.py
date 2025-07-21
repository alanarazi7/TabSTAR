import os.path
from dataclasses import dataclass, asdict
from os.path import join
from typing import Optional, Dict

import h5py
import numpy as np
from torch.utils.data import Dataset

from tabstar.tabstar_verbalizer import TabSTARData
from tabstar_paper.datasets.objects import SupervisedTask
from tabstar_paper.utils.io_handlers import dump_json, load_json


@dataclass
class DatasetProperties:
    name: str
    d_output: int
    idx2text: Dict[int, str]
    train_size: int
    val_size: int

    @classmethod
    def from_json(cls, data_dir: str) -> 'DatasetProperties':
        properties_path = join(data_dir, HDF5Dataset.PROPERTIES)
        d = load_json(properties_path)
        return cls(**d)

    @property
    def is_cls(self) -> bool:
        return self.d_output > 1

    @property
    def task_type(self) -> SupervisedTask:
        if self.d_output == 1:
            return SupervisedTask.REGRESSION
        elif self.d_output == 2:
            return SupervisedTask.BINARY
        elif self.d_output > 2:
            return SupervisedTask.MULTICLASS
        else:
            raise ValueError(f"Invalid d_output: {self.d_output}. Must be > 0.")


class HDF5Dataset(Dataset):

    TRAIN = "train"
    VAL = "val"
    X_TXT_KEY = "X_txt"
    X_NUM_KEY = "X_num"
    Y_KEY = "y"
    H5_FILE = "data.h5"
    PROPERTIES = "properties.json"

    def __init__(self, data_dir: str, is_train: bool):
        split_key = self.TRAIN if is_train else self.VAL
        self.file_path = join(data_dir, split_key, self.H5_FILE)
        self.properties: DatasetProperties = DatasetProperties.from_json(data_dir)
        self.size: int = self.properties.train_size if is_train else self.properties.val_size
        self.h5_file: Optional[h5py.File] = None

    def __len__(self) -> int:
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
    property_path = join(data_dir, HDF5Dataset.PROPERTIES)
    dump_json(asdict(properties), path=property_path)
    for data, split in [(train_data, HDF5Dataset.TRAIN), (val_data, HDF5Dataset.VAL)]:
        split_dir = join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        h5_file_path = join(split_dir, HDF5Dataset.H5_FILE)
        key2data = {
            HDF5Dataset.X_TXT_KEY: data.x_txt,
            HDF5Dataset.X_NUM_KEY: data.x_num,
            HDF5Dataset.Y_KEY: data.y
        }
        with h5py.File(h5_file_path, 'w') as h5f:
            for key, value in key2data.items():
                h5f.create_dataset(name=key, data=value)

