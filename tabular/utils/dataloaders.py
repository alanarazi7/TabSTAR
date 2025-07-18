from typing import List

from torch.utils.data import DataLoader


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

