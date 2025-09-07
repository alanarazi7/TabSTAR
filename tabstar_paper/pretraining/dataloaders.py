from random import sample, shuffle
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tabstar_paper.pretraining.hdf5 import HDF5Dataset
from tabstar_paper.pretraining.hyperparameters import TrainingArgs

# TODO: Make NUM_WORKERS configurable
NUM_WORKERS = 0

class MultiDatasetEpochBatches(Dataset):
    def __init__(self, datasets: List[HDF5Dataset], batch_size: int, max_samples_per_dataset: int):
        self.datasets = datasets
        self.batch_size = batch_size
        self.max_samples_per_dataset = max_samples_per_dataset
        self.batches = []
        self.make_batches()

    def make_batches(self):
        self.batches = []
        for dataset_idx, dataset in enumerate(self.datasets):
            indices = list(range(len(dataset)))
            if len(indices) >= self.max_samples_per_dataset:
                indices = sample(indices, self.max_samples_per_dataset)
            else:
                shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                self.batches.append((dataset_idx, batch_indices))
        shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        dataset_idx, batch_indices = self.batches[idx]
        batch = [self.datasets[dataset_idx][i] for i in batch_indices]
        return tabular_collate_fn(batch)


def get_dev_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    dataset = HDF5Dataset(data_dir=data_dir, is_train=False)
    return DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=batch_size, num_workers=0)


def tabular_collate_fn(batch):
    """We want to process the batch, so the (x, y) become np arrays, while only the first property returns.
    Here we assume that the properties are the same for all samples in the batch, i.e. we don't mix datasets."""
    x_txt, x_num, y, properties = zip(*batch)
    x_txt = np.array(x_txt)
    x_num = torch.tensor(np.array(x_num), dtype=torch.float32)
    properties = properties[0]
    y_dtype = torch.float32 if properties.d_output == 1 else torch.long
    y = torch.tensor(y, dtype=y_dtype)
    return x_txt, x_num, y, properties


def get_pretrain_multi_dataloader(data_dirs: List[str], args: TrainingArgs) -> DataLoader:
    datasets = [HDF5Dataset(data_dir=d, is_train=True) for d in data_dirs]
    multi_dataset = MultiDatasetEpochBatches(datasets=datasets, batch_size=args.batch_size,
                                             max_samples_per_dataset=args.epoch_examples)
    return DataLoader(multi_dataset, batch_size=None, num_workers=NUM_WORKERS)