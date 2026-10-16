from random import shuffle
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset, BatchSampler

from tabstarv1p1_pretrain.pretraining.data.dataloaders import pretrain_batch_collator
from tabstarv1p1_pretrain.pretraining.hdf5 import RealDataset
from tabstarv1p1_pretrain.utils.devices import get_cpu_cores
from tabstarv1p1_pretrain.pretraining.hyperparameters import MAX_DATASET_ROWS, TrainingArgs


ExampleKey = Tuple[int, int]
BatchKeys = List[ExampleKey]


class CombinedDataset(Dataset):
    def __init__(self, datasets: List[RealDataset]):
        self.datasets = datasets

    def __getitem__(self, key: Tuple[int, int]):
        ds_idx, item_idx = key
        return self.datasets[ds_idx][item_idx]

    def __len__(self) -> int:
        # Not used by DataLoader when driving via BatchSampler, but nice to have.
        return sum(len(ds) for ds in self.datasets)


class MixedBatchesSampler(BatchSampler):

    def __init__(self, datasets: List[RealDataset], train_args: TrainingArgs):
        self.datasets = datasets
        self.args = train_args
        self.batches: List[BatchKeys] = []
        self.make_batches()

    def make_batches(self):
        self.batches = []
        reg_batches = []
        cls_batches = []
        for dataset_idx, dataset in enumerate(self.datasets):
            indices = self._sample_dataset_indices(dataset=dataset)
            dataset_mini_batches = self._prepare_dataset_micro_batches(dataset_idx=dataset_idx, indices=indices)
            if dataset.is_cls:
                cls_batches.extend(dataset_mini_batches)
            else:
                reg_batches.extend(dataset_mini_batches)
        self._group_micro_batches(micro_batches=reg_batches)
        self._group_micro_batches(micro_batches=cls_batches)
        shuffle(self.batches)

    def _sample_dataset_indices(self, dataset: Dataset) -> List[int]:
        """Dropping the remainder keeps every micro-batch full, so gradient accumulation weighs
        every step's examples equally regardless of which dataset they came from.
        """
        assert len(dataset) <= MAX_DATASET_ROWS, f"{len(dataset)=} exceeds {MAX_DATASET_ROWS=}"
        dataset_indices = list(range(len(dataset)))
        shuffle(dataset_indices)
        usable = len(dataset_indices) - (len(dataset_indices) % self.args.micro_batch_size)
        return dataset_indices[:usable]

    def _prepare_dataset_micro_batches(self, dataset_idx: int, indices: List[int]) -> List[BatchKeys]:
        mini_batches = []
        for i in range(0, len(indices), self.args.micro_batch_size):
            batch_indices = indices[i:i + self.args.micro_batch_size]
            batch_keys = [(dataset_idx, idx) for idx in batch_indices]
            mini_batches.append(batch_keys)
        return mini_batches

    def _group_micro_batches(self, micro_batches: List[BatchKeys]):
        shuffle(micro_batches)
        num_micro_batches = self.args.batch_size // self.args.micro_batch_size
        for i in range(0, len(micro_batches), num_micro_batches):
            our_batches = micro_batches[i:i + num_micro_batches]
            keys = [key for mb in our_batches for key in mb]
            self.batches.append(keys)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for i, batches in enumerate(self.batches):
            yield batches


def get_mixed_dataloader(datasets: List[RealDataset], train_args: TrainingArgs) -> DataLoader:
    combined_dataset = CombinedDataset(datasets=datasets)
    sampler = MixedBatchesSampler(datasets=combined_dataset.datasets, train_args=train_args)
    # Capped at 8: an uncapped worker count can exhaust file descriptors on a many-core server.
    num_workers = min(get_cpu_cores(), 8)
    return DataLoader(combined_dataset, batch_sampler=sampler, collate_fn=pretrain_batch_collator,
                      num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

