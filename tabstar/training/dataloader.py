from pandas import DataFrame, Series
from torch.utils.data import Dataset, DataLoader

from tabstar.arch.config import BATCH_SIZE


class PandasDataset(Dataset):
    def __init__(self, x: DataFrame, y: Series):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x.iloc[idx], self.y.iloc[idx]
    
def get_dataloader(x: DataFrame, y: Series) -> DataLoader:
    dataset = PandasDataset(x=x, y=y)
    return DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=0, collate_fn=collate_fn)


def collate_fn(batch):
    x_batch = [item[0] for item in batch]
    y_batch = [item[1] for item in batch]
    return x_batch, y_batch