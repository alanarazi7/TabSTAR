from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

from tabstar.arch.config import LORA_LR

WARMUP_PROPORTION = 0.1
MAX_EPOCHS = 50


def get_optimizer(model: nn.Module) -> AdamW:
    params = [{"params": model.parameters(), "lr": LORA_LR, "name": "lora_lr"}]
    optimizer = AdamW(params)
    return optimizer

def get_scheduler(optimizer: AdamW) -> LRScheduler:
    return OneCycleLR(optimizer=optimizer, max_lr=LORA_LR, total_steps=MAX_EPOCHS,
                      pct_start=WARMUP_PROPORTION, anneal_strategy='cos')