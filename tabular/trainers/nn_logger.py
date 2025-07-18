import wandb
from torch.optim.lr_scheduler import LRScheduler


def log_general(scheduler: LRScheduler, steps: int, epoch: int):
    ret = {'Steps': steps}
    for param_grp, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
        ret[f"LR {param_grp['name']}"] = lr
    wandb.log(ret, step=epoch)
