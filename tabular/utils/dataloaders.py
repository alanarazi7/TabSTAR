from os.path import join
from typing import List

import numpy as np
import torch
from torch.utils.data import Subset, Dataset, DataLoader

from tabular.datasets.raw_dataset import MAX_DATASET_EXAMPLES
from tabular.datasets.torch_dataset import HDF5Dataset
from tabular.preprocessing.splits import DataSplit
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs, MAX_EPOCH_EXAMPLES
from tabular.utils.utils import verbose_print


def get_pretrain_epoch_dataloader(data_dirs: List[str], batch_size: int):
    verbose_print(f"🚜 Pretrain epoch dataloader with {len(data_dirs)} datasets and batch size {batch_size}")
    datasets = [HDF5Dataset(split_dir=join(d, DataSplit.TRAIN)) for d in data_dirs]
    subset_datasets = [get_subset_dataset(dataset=d) for d in datasets]
    dataloaders = [DataLoader(ds, collate_fn=tabular_collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
                   for ds in subset_datasets]
    return dataloaders


def round_robin_batches(dataloaders: List[DataLoader]):
    """
    Yields batches from each DataLoader in a round-robin manner until all are exhausted.
    Each yielded element is a tuple (dataset_index, batch), so you know which dataset
    the batch came from.
    """
    iterators = [iter(dl) for dl in dataloaders]
    active = [True] * len(iterators)

    while any(active):
        for i, it in enumerate(iterators):
            if not active[i]:
                continue
            try:
                batch = next(it)
                yield batch
            except StopIteration:
                active[i] = False

def get_dataloader(data_dir: str, split: DataSplit, batch_size: int) -> DataLoader:
    dataset = HDF5Dataset(split_dir=join(data_dir, split))
    assert split != DataSplit.TRAIN
    return DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=batch_size, num_workers=0)


def get_subset_dataset(dataset: Dataset):
    # Make sure we don't try to sample more examples than exist in the dataset
    num_samples = min(MAX_EPOCH_EXAMPLES, len(dataset))
    # Generate a random permutation of indices and select the first num_samples indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    # Return a lazy subset (only the indices are stored; actual data is loaded on-demand)
    return Subset(dataset, indices)



def tabular_collate_fn(batch):
    """We want to process the batch, so the (x, y) become np arrays, while only the first property returns.
    Here we assume that the properties are the same for all samples in the batch, i.e. we don't mix datasets."""
    x_txt, x_num, y, properties = zip(*batch)
    x_txt = np.array(x_txt)
    x_num = np.array(x_num)
    y = np.array(y)
    properties = properties[0]
    return x_txt, x_num, y, properties
