from torch import nn
from torch.optim import AdamW


def get_lora_optimizer(model: nn.Module, lr: float, wd: float = 0.0) -> AdamW:
    params = [{"params": model.parameters(), "lr": lr, "name": "lora_lr", "weight_decay": wd}]
    optimizer = AdamW(params)
    return optimizer