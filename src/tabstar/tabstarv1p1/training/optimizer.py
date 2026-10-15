from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

WARMUP_PROPORTION = 0.1


def get_epoch_scheduler(optimizer: AdamW, max_lr: float, epochs: int) -> LRScheduler:
    """OneCycle measured in epochs, for a caller that steps it once per epoch."""
    # TODO: we currently use total_steps=epochs, but it expects to be affected by accumulation steps.
    # We might want to reconsider setting this depending on the expected number of updates.
    try:
        return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=epochs,
                          pct_start=WARMUP_PROPORTION, anneal_strategy='cos')
    except ZeroDivisionError:
        return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=epochs+1,
                          pct_start=WARMUP_PROPORTION, anneal_strategy='cos')


def get_step_scheduler(optimizer: AdamW, max_lr: float, steps_per_epoch: int, epochs: int) -> LRScheduler:
    """OneCycle measured in optimizer updates, for a caller that steps it on every update."""
    total_steps = max(steps_per_epoch * epochs, 2)
    return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=total_steps,
                      pct_start=WARMUP_PROPORTION, anneal_strategy='cos')
