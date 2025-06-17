import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from tabstar.arch.config import BATCH_SIZE


from torch.utils.data import Dataset

from tabstar.tabstar_verbalizer import TabSTARData


class TabSTARDataset(Dataset):
    def __init__(self, data: TabSTARData):
        self.x_txt = data.x_txt
        self.x_num = data.x_num
        if isinstance(data.y, pd.Series):
            self.y = data.y.reset_index(drop=True)
        elif isinstance(data.y, np.ndarray):
            self.y = pd.Series(data.y, dtype=np.float32)
        else:
            # Dummy target for convenience
            self.y = pd.Series(np.zeros(len(data.x_txt)), dtype=np.float32)
        self.d_output = data.d_output

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x_txt = self.x_txt[idx]
        x_num = self.x_num[idx]
        y = self.y.iloc[idx]
        return x_txt, x_num, y, self.d_output
    
def get_dataloader(data: TabSTARData, is_train: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    dataset = TabSTARDataset(data)
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)


def collate_fn(batch) -> TabSTARData:
    x_txt_batch, x_num_batch, y_batch, d_output_batch = zip(*batch)
    # Assuming all batches have the same d_output, which is correct for finetune
    d_output = d_output_batch[0]
    x_txt = np.stack(x_txt_batch)
    x_num = np.stack(x_num_batch)
    y = pd.Series(y_batch)
    data = TabSTARData(d_output=d_output, x_txt=x_txt, x_num=x_num, y=y)
    return data