from dataclasses import dataclass, asdict

import torch
from torch import nn, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tabstar.training.early_stopping import EarlyStopping
from tabstar_paper.pretraining.seeds import Seeds


@dataclass
class Checkpoint:
    model_dict: dict
    optimizer_dict: dict
    scheduler_dict: dict
    scaler_dict: dict
    early_stop_dict: dict
    seeds_dict: dict
    epoch: int
    steps: int



def save_checkpoint(save_path: str, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, scaler: GradScaler,
                    early_stopping: EarlyStopping, epoch: int, steps: int):
    cp = Checkpoint(
        model_dict=model.state_dict(),
        optimizer_dict=optimizer.state_dict(),
        scheduler_dict=scheduler.state_dict(),
        scaler_dict=scaler.state_dict(),
        early_stop_dict=early_stopping.state_dict(),
        seeds_dict=Seeds.state_dict(),
        epoch=epoch,
        steps=steps,
    )
    d = asdict(cp)
    torch.save(d, save_path)


def load_checkpoint(load_path: str, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler,
                    scaler: GradScaler, early_stopping: EarlyStopping) -> Checkpoint:
    cp = torch.load(load_path)
    cp = Checkpoint(**cp)
    model.load_state_dict(cp.model_dict)
    optimizer.load_state_dict(cp.optimizer_dict)
    scheduler.load_state_dict(cp.scheduler_dict)
    scaler.load_state_dict(cp.scaler_dict)
    early_stopping.load_state_dict(cp.early_stop_dict)
    Seeds.load_state_dict(cp.seeds_dict)
    return cp