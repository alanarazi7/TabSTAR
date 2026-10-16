from typing import Optional, Dict

import wandb


def log_epoch(train_loss: float, dev_loss: float, metric: float, epoch: int, other_args: Optional[Dict] = None):
    if wandb.run is None:
        # Callers outside do_pretrain.py never call wandb.init(); this logging is optional.
        return
    d = {'train_loss': train_loss, 'dev_loss': dev_loss, 'metric': metric, 'epoch': epoch}
    for k, v in (other_args or {}).items():
        d[k] = v
    wandb.log(d, step=epoch)