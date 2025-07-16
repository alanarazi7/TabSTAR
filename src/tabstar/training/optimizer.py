from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

from tabstar.arch.config import LORA_LR

WARMUP_PROPORTION = 0.1
MAX_EPOCHS = 50


def get_optimizer(model: nn.Module, lr: float = LORA_LR) -> AdamW:
    params = [{"params": model.parameters(), "lr": lr, "name": "lora_lr"}]
    optimizer = AdamW(params)
    return optimizer

def get_scheduler(optimizer: AdamW, max_lr: float = LORA_LR, epochs: int = MAX_EPOCHS) -> LRScheduler:
    return OneCycleLR(optimizer=optimizer, max_lr=max_lr, total_steps=epochs,
                      pct_start=WARMUP_PROPORTION, anneal_strategy='cos')