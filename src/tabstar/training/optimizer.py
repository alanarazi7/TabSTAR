from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

WARMUP_PROPORTION = 0.1


def get_optimizer(model: nn.Module, lr: float, wd: float) -> AdamW:
    params = [{"params": model.parameters(), "lr": lr, "wd": wd}]
    optimizer = AdamW(params)
    return optimizer

def get_scheduler(optimizer: AdamW, max_lr: float, epochs: int) -> LRScheduler:
    # TODO: we currently use total_steps=epochs, but it expects to be affected by accumulation steps.
    # We might want to reconsider setting this depending on the expected number of updates.
    try:
        return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=epochs,
                          pct_start=WARMUP_PROPORTION, anneal_strategy='cos')
    except ZeroDivisionError:
        return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=epochs+1,
                          pct_start=WARMUP_PROPORTION, anneal_strategy='cos')