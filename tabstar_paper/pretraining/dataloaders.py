from random import sample, shuffle
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tabstar.preprocessing.tokenization import tokenize
from tabstar.training.devices import CPU_CORES
from tabstar_paper.pretraining.hdf5 import HDF5Dataset
from tabstar_paper.pretraining.hyperparameters import MAX_EPOCH_EXAMPLES

# TODO: Make NUM_WORKERS configurable
NUM_WORKERS = CPU_CORES
PIN_MEMORY = False
PERSISTENT_WORKERS = False

class MultiDatasetEpochBatches(Dataset):
    def __init__(self, datasets: List[HDF5Dataset], batch_size: int, max_samples_per_dataset: int = MAX_EPOCH_EXAMPLES):
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
    return DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=batch_size,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)


def tabular_collate_fn(batch):
    x_txt_batch, x_num_batch, y_batch, properties_batch = zip(*batch)
    x_num = torch.tensor(np.array(x_num_batch), dtype=torch.float32)
    assert len({p.name for p in properties_batch}) == 1, "All samples in the batch must come from the same dataset."
    properties = properties_batch[0]
    d_output = properties.d_output
    y_dtype = torch.float32 if d_output == 1 else torch.long
    y = torch.tensor(y_batch, dtype=y_dtype)

    # For x_txt, we convert to indices and tokenize the unique texts
    x_txt_flat = np.array(x_txt_batch).reshape(-1)
    unique_texts, inverse_indices = np.unique(x_txt_flat, return_inverse=True)
    x_txt = inverse_indices.reshape(len(x_txt_batch), -1)
    x_txt = torch.tensor(x_txt, dtype=torch.long)
    assert x_txt.shape == x_num.shape
    tokenized = tokenize(texts=list(unique_texts))

    return {
        'x_txt': x_txt,
        'tokenized': tokenized,
        'x_num': x_num,
        'y': y,
        'd_output': d_output,
        'dataset_name': properties.name,
    }


def get_pretrain_multi_dataloader(data_dirs: List[str], batch_size: int) -> DataLoader:
    datasets = [HDF5Dataset(data_dir=d, is_train=True) for d in data_dirs]
    multi_dataset = MultiDatasetEpochBatches(datasets=datasets, batch_size=batch_size)
    return DataLoader(multi_dataset, batch_size=None, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                      persistent_workers=PERSISTENT_WORKERS)