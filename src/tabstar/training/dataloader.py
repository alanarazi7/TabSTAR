import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tabstar.arch.config import BATCH_SIZE
from tabstar.preprocessing.tokenization import tokenize
from tabstar.tabstar_verbalizer import TabSTARData


class TabSTARDataset(Dataset):
    def __init__(self, data: TabSTARData):
        self.x_txt = data.x_txt
        self.x_num = data.x_num
        self.d_output = data.d_output
        dtype = torch.float32 if self.d_output == 1 else torch.int64
        if isinstance(data.y, pd.Series):
            self.y = torch.tensor(data.y.values, dtype=dtype)
        elif isinstance(data.y, np.ndarray):
            self.y = torch.tensor(data.y, dtype=dtype)
        else:
            self.y = torch.zeros(len(data.x_txt), dtype=dtype)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        # x_txt stays as string (tokenize in collate_fn)
        x_txt = self.x_txt[idx]
        x_num = torch.tensor(self.x_num[idx], dtype=torch.float32)
        y = self.y[idx]
        return x_txt, x_num, y, self.d_output


def get_dataloader(data: TabSTARData, is_train: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    dataset = TabSTARDataset(data)
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)


def collate_fn(batch):
    x_txt_batch, x_num_batch, y_batch, d_output_batch = zip(*batch)
    y = torch.stack(y_batch)
    x_num = torch.stack(x_num_batch)
    d_output = d_output_batch[0]
    assert set(d_output_batch) == {d_output}, "All items in the batch must have the same d_output value"

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
        'd_output': d_output
    }