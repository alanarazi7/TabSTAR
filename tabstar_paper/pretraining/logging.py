from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import wandb


def summarize_model(model: nn.Module):
    m_total_params = sum(p.numel() for p in model.parameters()) / 1000000
    m_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    print(f"Total parameters: {m_total_params:.2f}M. Trainable: {m_trainable:.2f}M")
    for name, submodule in model.named_children():
        submodule_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        print(f"{name}: {submodule_params:,} parameters")
    if hasattr(model, 'text_encoder'):
        for name, submodule in model.text_encoder.named_children():
            submodule_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            total_submodule_params = sum(p.numel() for p in submodule.parameters())
            print(f"Text encoder {name}: {submodule_params:,}/{total_submodule_params:,} trained parameters")

def log_epoch_start(scheduler: LRScheduler, steps: int, epoch: int):
    ret = {'Steps': steps}
    for param_grp, lr in zip(scheduler.optimizer.param_groups, scheduler.get_last_lr()):
        ret[f"LR {param_grp['name']}"] = lr
    wandb.log(ret, step=epoch)