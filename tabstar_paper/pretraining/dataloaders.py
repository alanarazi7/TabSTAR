import numpy as np
from torch.utils.data import DataLoader

from tabstar_paper.pretraining.hdf5 import HDF5Dataset


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