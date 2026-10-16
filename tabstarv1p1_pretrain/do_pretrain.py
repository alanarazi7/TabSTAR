import argparse
import resource
from dataclasses import asdict
from os.path import exists

import wandb

from tabstar.tabstarv1p1.training.devices import get_device

from tabstarv1p1_pretrain.pretraining.arguments import REG_DIST_BINS
from tabstarv1p1_pretrain.pretraining.pretrain_args import PretrainArgs, PretrainConfig, TaskType, PoolMode, TEXT_BATCH_SIZE
from tabstarv1p1_pretrain.finetunability.args import FinetunabilityArgs
from tabstarv1p1_pretrain.pretraining.pretrainer import TabSTARPretrainer
from tabstarv1p1_pretrain.utils.devices import safe_gpu_training
from tabstarv1p1_pretrain.utils.logging import git_info
from tabstarv1p1_pretrain.constants import DEVICE
from tabstarv1p1_pretrain.pretraining.hyperparameters import (TEXTUAL_UNFREEZE_LAYERS, LR, WARMUP,
                                                       WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE,
                                                       TrainingArgs, MICRO_BATCH_SIZE)
from tabstarv1p1_pretrain.utils.logging import wandb_run, wandb_finish


def raise_fd_limit() -> None:
    """DataLoader workers each hold an fd open per in-flight tensor (default `file_descriptor`
    sharing strategy); num_workers * prefetch_factor can exceed the default 1024 soft
    RLIMIT_NOFILE and deadlock the DataLoader on EMFILE instead of raising. Raise the soft
    limit to the hard ceiling before any workers are spawned.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def do_pretrain(train_args: TrainingArgs,
                pretrain_args: PretrainArgs,
                finetunability_args: FinetunabilityArgs,
                wandb_project: str = "tabstarv1p1_pretrain") -> str:
    if exists(pretrain_args.path):
        print(f"Pretraining model already exists for {pretrain_args.raw_exp_name}")
        return
    device = get_device(device=DEVICE)
    wandb_run(exp_name=pretrain_args.raw_exp_name, project=wandb_project, tags=[pretrain_args.task_type])
    d_summary = {**asdict(pretrain_args), **asdict(train_args), **asdict(finetunability_args),
                'full_exp_name': pretrain_args.full_exp_name}
    wandb.config.update(d_summary, allow_val_change=True)
    model = TabSTARPretrainer(run_name=pretrain_args.full_exp_name,
                              device=device,
                              train_args=train_args,
                              pretrain_args=pretrain_args,
                              finetunability_args=finetunability_args)
    print(f"🧪 [{git_info()}] Initializing exp {pretrain_args.full_exp_name} "
          f"over {model.num_t4_datasets_used} T4 datasets ({pretrain_args.task_type})")
    pretrain_args.num_datasets = model.num_t4_datasets_used
    pretrain_args.t4_pool_sizes = model.t4_pool_sizes
    wandb.config.update({'num_datasets': pretrain_args.num_datasets, 't4_pool_sizes': pretrain_args.t4_pool_sizes},
                        allow_val_change=True)
    PretrainConfig(pretrain_args=pretrain_args, train_args=train_args,
                  finetunability_args=finetunability_args).to_json()
    model.train()
    print(f"🌟 TabSTAR was pretrained. The experiment name is: {pretrain_args.full_exp_name}")
    wandb_finish(d_summary)
    return pretrain_args.full_exp_name



if __name__ == "__main__":
    raise_fd_limit()

    parser = argparse.ArgumentParser(description="Run the training script with optional arguments.")
    # General
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default="tabstarv1p1_pretrain")
    # Arch
    parser.add_argument('--e5_unfreeze_layers', type=int, default=TEXTUAL_UNFREEZE_LAYERS)
    parser.add_argument('--text_batch_size', type=int, default=TEXT_BATCH_SIZE)
    # Data
    parser.add_argument('--t4_dir', type=str, default=None,
                        help="Dir of pre-built T4 RealDataset dirs (from do_t4_data.py) to add to pretraining.")
    parser.add_argument('--num_datasets', type=int, default=1000,
                        help="Cap on how many T4 datasets to sample from --t4_dir.")
    parser.add_argument('--task_type', choices=list(TaskType), required=True,
                        help="Restrict T4 datasets (and finetunability tracking) to one task type.")
    parser.add_argument('--pool_mode', choices=list(PoolMode), default=PoolMode.ACTIVE,
                        help="How --num_datasets get drawn from --t4_dir: 'random' samples once at "
                             "startup and keeps that subset for the whole run; 'active' resamples a "
                             "fresh random subset from the full pool at the start of every epoch; "
                             "'first' deterministically takes the first --num_datasets dirs in sorted "
                             "order and keeps that fixed subset for the whole run.")
    parser.add_argument('--reg_dist_bins', type=int, default=REG_DIST_BINS,
                        help="Number of bins for the regression head's distributional sub-head.")
    # Training
    parser.add_argument('--learning_rate', type=float, default=LR)
    parser.add_argument('--warmup', type=float, default=WARMUP)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--micro_batch_size', type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    # Finetunability tracking
    parser.add_argument('--finetunability_timeout_s', dest='timeout_s', type=float, default=1800.0)
    parser.add_argument('--finetunability_skip_epoch_zero', dest='skip_epoch_zero',
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Skip the finetunability check before the first pretrain epoch.")

    args = parser.parse_args()
    train = TrainingArgs.from_args(args)
    pretrain = PretrainArgs.from_args(args=args)
    finetunability = FinetunabilityArgs.from_args(args)
    with safe_gpu_training():
        full_exp_name = do_pretrain(train_args=train, pretrain_args=pretrain,
                                    finetunability_args=finetunability, wandb_project=args.wandb_project)
    # Printing this for retrieval of automatic pipeline
    print(full_exp_name)
