from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tabstar_paper.pretraining.hdf5 import HDF5Dataset
from tabstar_paper.pretraining.hyperparameters import MAX_EPOCH_EXAMPLES


def get_dev_dataloader(data_dir: str, batch_size: int) -> DataLoader:
    dataset = HDF5Dataset(data_dir=data_dir, is_train=False)
    return DataLoader(dataset, shuffle=False, collate_fn=tabular_collate_fn, batch_size=batch_size, num_workers=0)


def tabular_collate_fn(batch):
    """We want to process the batch, so the (x, y) become np arrays, while only the first property returns.
    Here we assume that the properties are the same for all samples in the batch, i.e. we don't mix datasets."""
    x_txt, x_num, y, properties = zip(*batch)
    x_txt = np.array(x_txt)
    x_num = np.array(x_num)
    y = np.array(y)
    properties = properties[0]
    return x_txt, x_num, y, properties


def get_pretrain_epoch_dataloader(data_dirs: List[str], batch_size: int) -> List[DataLoader]:
    dataloaders = []
    for d in data_dirs:
        dataset = HDF5Dataset(data_dir=d, is_train=True)
        subset_dataset = get_subset_dataset(dataset=dataset)
        dataloader = DataLoader(subset_dataset, collate_fn=tabular_collate_fn, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        dataloaders.append(dataloader)
    return dataloaders


def get_subset_dataset(dataset: Dataset):
    # Make sure we don't try to sample more examples than exist in the dataset
    num_samples = min(MAX_EPOCH_EXAMPLES, len(dataset))
    # Generate a random permutation of indices and select the first num_samples indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    # Return a lazy subset (only the indices are stored; actual data is loaded on-demand)
    return Subset(dataset, indices)