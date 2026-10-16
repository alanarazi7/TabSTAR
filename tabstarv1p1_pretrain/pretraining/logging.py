from typing import Dict, Optional

import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from tabstarv1p1_pretrain.pretraining.monitoring import gpu_mem_gb, gpu_contention, peak_gpu_mem_gb


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


def log_epoch_metrics(epoch: int, train_loss: float, elapsed: float, train_minutes: float, device: torch.device):
    log_dict = {'train_loss': train_loss,
               'all/total_train_loss': train_loss,
               'elapsed': elapsed,
               'train_minutes': train_minutes,
               'memory': gpu_mem_gb(device),
               'peak_memory': peak_gpu_mem_gb(device)}
    contention = gpu_contention(device)
    if contention is not None:
        log_dict.update({f"gpu/{k}": v for k, v in contention.items()})
    wandb.log(log_dict, step=epoch)


def print_epoch_summary(epoch: int, train_minutes: float, total_minutes: float, train_loss: float, peak_mem_gb: float):
    print(f"Epoch {epoch} || Train {train_minutes:.1f}m (total {total_minutes:.1f}m) || "
          f"Loss {train_loss:.3f} || Peak GPU {peak_mem_gb:.2f}GB")


def log_finetunability(epoch: int, scores: Dict[str, float], minutes: float,
                       best_scores: Optional[Dict[str, float]] = None):
    def fmt(key: str) -> str:
        value = scores.get(key)
        return f"{value:.0f}" if value is not None else "n/a"

    # Elo only in the console print -- the per-dataset scores (one entry each, too many to
    # summarize on one line) are still logged to WandB below.
    best_elo = (best_scores or {}).get("elo")
    best_suffix = f" || best {best_elo:.0f}" if best_elo is not None else ""
    print(f"🏟️ Epoch {epoch}: Finetunability ({minutes:.1f}m) || "
          f"{fmt('elo')} (cls {fmt('cls_elo')} / reg {fmt('reg_elo')}){best_suffix}")
    # Per-dataset scores get their own panel group, separate from the aggregate means -- there are
    # 51 of them, too many to sit alongside the main metrics.
    log_dict = {}
    for key, value in scores.items():
        if key.startswith("dataset_"):
            log_dict[f"finetune_datasets/{key[len('dataset_'):]}"] = value
        elif key.startswith("mean_"):
            continue
        elif key in ("elo", "cls_elo", "reg_elo"):
            log_dict[f"val_finetune_{key}"] = value
    for key in ("cls_elo", "reg_elo"):
        if key in scores:
            log_dict[f"tasks/{key}"] = scores[key]
            log_dict[f"all/finetune_{key}"] = scores[key]
    full_elo = scores.get("elo")
    if full_elo is not None:
        log_dict["all/finetunability_elo"] = full_elo
    for key, value in (best_scores or {}).items():
        if key == "elo":
            log_dict["all/finetunability_elo_best"] = value
        elif key in ("cls_elo", "reg_elo"):
            log_dict[f"all/finetune_{key}_best"] = value
    wandb.log(log_dict, step=epoch)


def log_training_complete(epoch: int):
    wandb.log({'train_epochs': epoch})
