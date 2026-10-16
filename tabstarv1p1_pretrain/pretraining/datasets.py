from typing import List

from torch.utils.data import DataLoader

from tabstarv1p1_pretrain.pretraining.hdf5 import RealDataset
from tabstarv1p1_pretrain.training.dataloaders import get_mixed_dataloader
from tabstarv1p1_pretrain.pretraining.hyperparameters import TrainingArgs


def get_pretrain_multi_dataloader(data_dirs: List[str], train_args: TrainingArgs) -> DataLoader:
    datasets = []
    for d in data_dirs:
        dataset = RealDataset(data_dir=d)
        datasets.append(dataset)
    return get_mixed_dataloader(datasets=datasets, train_args=train_args)
