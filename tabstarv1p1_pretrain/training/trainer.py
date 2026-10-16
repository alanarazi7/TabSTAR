import torch
from torch.nn import Module
from torch.optim import Optimizer

from tabstarv1p1_pretrain.training.monitoring import log_grad_stats, log_param_stats
from tabstarv1p1_pretrain.constants import MONITOR


def update_model(model: Module, optimizer: Optimizer):
    if MONITOR:
        log_grad_stats(model=model, step="before_clip")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if MONITOR:
        log_grad_stats(model=model, step="after_clip")
    optimizer.step()
    if MONITOR:
        log_param_stats(model=model, step="after_update")
    optimizer.zero_grad()