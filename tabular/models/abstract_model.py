from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tabular.datasets.tabular_datasets import OpenMLDatasetID
from tabular.datasets.properties import DatasetProperties
from tabular.datasets.torch_dataset import get_data_dir, get_properties
from tabular.evaluation.predictions import Predictions
from tabular.preprocessing.splits import DataSplit
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask
from tabular.preprocessing.trees.categorical import ColumnLabelEncoder
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.utils import fix_seed


class TabularModel:

    MODEL_NAME: str
    SHORT_NAME: str
    PROCESSING: PreprocessingMethod

    def __init__(self, run_name: str, dataset_ids: List[OpenMLDatasetID], device: torch.device,
                 run_num: int, train_examples: int = 0, args: Optional[PretrainArgs] = None,
                 carte_lr_index: Optional[int] = None):
        fix_seed()
        self.run_name = run_name
        self.dataset_ids = dataset_ids
        self.device = device
        self.run_num = run_num
        self.train_examples = train_examples
        self.args = args
        self.data_dirs: List[str] = self.initialize_data_dirs()
        self.datasets: List[DatasetProperties] = [get_properties(d) for d in self.data_dirs]
        self.model: Optional[Any] = None
        self.carte_lr_index = carte_lr_index
        self.config = self.set_config()
        # For processing
        self.y_scaler: Optional[StandardScaler] = None
        self.x_median: Optional[Dict[str, float]] = None
        self.x_encoder: Optional[Dict[str, ColumnLabelEncoder]] = None

    def initialize_data_dirs(self) -> List[str]:
        data_dirs = []
        for d in tqdm(self.dataset_ids, desc="Initializing data dirs", leave=False):
            if isinstance(self.args, PretrainArgs):
                number_verbalization = self.args.numbers_verbalization
            elif isinstance(self.args, FinetuneArgs):
                number_verbalization = self.args.pretrain_args.numbers_verbalization
            else:
                number_verbalization = None
            data = get_data_dir(dataset=d, processing=self.PROCESSING, run_num=self.run_num,
                                train_examples=self.train_examples, device=self.device,
                                number_verbalization=number_verbalization)
            data_dirs.append(data)
        return data_dirs

    @property
    def dataset(self) -> DatasetProperties:
        if len(self.datasets) > 1:
            raise ValueError("Multiple datasets found! This property shouldn't be used during pretraining")
        return self.datasets[0]

    @property
    def data_dir(self) -> str:
        if len(self.data_dirs) > 1:
            raise ValueError("Multiple data directories found! This property shouldn't be used during pretraining")
        return self.data_dirs[0]

    @property
    def task_type(self) -> SupervisedTask:
        return self.dataset.task_type

    def set_config(self) -> Optional[dataclass]:
        raise NotImplementedError("Default config method not implemented yet")

    def initialize_model(self):
        raise NotImplementedError("Initialize model method not implemented yet")

    def train(self):
        raise NotImplementedError("Train method not implemented yet")

    def test(self) -> Dict[DataSplit, Predictions]:
        raise NotImplementedError("Test method not implemented yet")

