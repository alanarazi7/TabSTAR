from typing import Dict

import wandb
from torch.optim.lr_scheduler import LRScheduler

from tabstar.training.early_stopping import EarlyStopping
from tabular.datasets.properties import DatasetProperties
from tabular.evaluation.loss import LossAccumulator
from tabular.evaluation.predictions import Predictions


def log_general(scheduler: LRScheduler, steps: int, epoch: int):
    ret = {'Steps': steps}
    for param_grp, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
        ret[f"LR {param_grp['name']}"] = lr
    wandb.log(ret, step=epoch)


def log_dev_loss(dev_loss: LossAccumulator, epoch: int):
    wandb.log({f'All Downstream/val_loss': dev_loss.avg}, step=epoch)

def log_dev_performance(properties: DatasetProperties, epoch: int,
                        data_dev_loss: LossAccumulator, predictions: Predictions):
    cat = f"Downstream/{properties.sid}"
    wandb.log({f'{cat}/val_loss': data_dev_loss.avg, f'{cat}/val_metric': predictions.score}, step=epoch)


def log_train_loss(train_loss: LossAccumulator, epoch: int, dataset2losses: Dict[str, LossAccumulator]):
    wandb.log({f'All Downstream/train_loss': train_loss.avg}, step=epoch)
    for sid, data_train_loss in dataset2losses.items():
        wandb.log({f'Downstream/{sid}/train_loss': data_train_loss.avg}, step=epoch)

def summarize_epoch(epoch: int, train_loss: LossAccumulator, dev_loss: LossAccumulator, metric_score: float,
                    early_stopper: EarlyStopping):
    log_str = f"Epoch {epoch} || Train {train_loss.avg} || Val {dev_loss.avg} || Metric {metric_score:.4f}"
    if metric_score > early_stopper.metric:
        log_str += " 🥇"
    else:
        log_str += f" 😓 [{early_stopper.failed}]"
    print(log_str)

def prefix(is_pretrain: bool) -> str:
    return 'Pretrain' if is_pretrain else 'Downstream'
