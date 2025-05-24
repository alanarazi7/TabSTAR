from typing import Dict

import wandb
from torch.optim.lr_scheduler import LRScheduler

from tabular.datasets.properties import DatasetProperties
from tabular.evaluation.loss import LossAccumulator
from tabular.evaluation.predictions import Predictions
from tabular.utils.early_stopping import EarlyStopping
from tabular.utils.utils import cprint


def log_general(scheduler: LRScheduler, steps: int, epoch: int):
    ret = {'Steps': steps}
    for param_grp, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
        ret[f"LR {param_grp['name']}"] = lr
    wandb.log(ret, step=epoch)


def log_dev_loss(is_pretrain: bool, dev_loss: LossAccumulator, metric: float, epoch: int):
    cat = prefix(is_pretrain)
    wandb.log({f'All {cat}/val_loss': dev_loss.avg}, step=epoch)
    if is_pretrain:
        wandb.log({f'All {cat}/val_metric': metric}, step=epoch)

def log_dev_performance(properties: DatasetProperties, is_pretrain: bool, epoch: int,
                        data_dev_loss: LossAccumulator, predictions: Predictions):
    cat = f"{prefix(is_pretrain)}/{properties.sid}"
    wandb.log({f'{cat}/val_loss': data_dev_loss.avg, f'{cat}/val_metric': predictions.score}, step=epoch)


def log_train_loss(train_loss: LossAccumulator, epoch: int, is_pretrain: bool, dataset2losses: Dict[str, LossAccumulator]):
    cat = f"{prefix(is_pretrain)}"
    wandb.log({f'All {cat}/train_loss': train_loss.avg}, step=epoch)
    for sid, data_train_loss in dataset2losses.items():
        wandb.log({f'{cat}/{sid}/train_loss': data_train_loss.avg}, step=epoch)

def summarize_epoch(epoch: int, train_loss: LossAccumulator, dev_loss: LossAccumulator, metric_score: float,
                    early_stopper: EarlyStopping, is_pretrain: bool):
    log_str = f"Epoch {epoch} || Train {train_loss.avg} || Val {dev_loss.avg} || Metric {metric_score:.4f}"
    if metric_score > early_stopper.metric:
        log_str += " 🥇"
    elif is_pretrain:
        log_str += f" 😓 [{early_stopper.epochs_without_improvement}]"
    cprint(log_str)

def prefix(is_pretrain: bool) -> str:
    return 'Pretrain' if is_pretrain else 'Downstream'
