import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData
from tabstar.tabstarv1p1.training.hyperparams import GLOBAL_BATCH
from tabstar.tabstarv1p1.finetune.collator import pretrain_batch_collator


class DownstreamDataset(Dataset):
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
        return x_num, x_txt, y, self.d_output
    
def get_dataloader(data: TabSTARData, is_train: bool, batch_size: int = GLOBAL_BATCH) -> DataLoader:
    dataset = DownstreamDataset(data)
    # 0: small in-memory data, no I/O to parallelize; workers here only add overhead and exhaust
    # resources when spun up fresh per dataset.
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size, collate_fn=pretrain_batch_collator,
                      num_workers=0, pin_memory=True)
