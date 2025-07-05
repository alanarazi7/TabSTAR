from dataclasses import asdict
from os.path import exists
from typing import List

import wandb

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.training.devices import get_device
from tabstar_paper.constants import GPU
from tabstar_paper.pretraining.pretrainer import TabSTARPretrainer
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.logging import wandb_run, RunType
from tabular.utils.utils import cprint


def do_pretrain(pretrain_datasets: List[TabularDatasetID], pretrain_args: PretrainArgs):
    if exists(pretrain_args.path):
        print(f"Pretraining model already exists for {pretrain_args.full_exp_name}")
        return
    cprint(f"🧪 Initializing experiment {pretrain_args.full_exp_name}")
    device = get_device(device=GPU)
    wandb_run(exp_name=pretrain_args.raw_exp_name, run_type=RunType.PRETRAIN)
    wandb.config.update(asdict(pretrain_args), allow_val_change=True)
    cprint(f"Pretraining over {len(pretrain_datasets)} datasets")
    model = TabSTARPretrainer(run_name=pretrain_args.full_exp_name, dataset_ids=pretrain_datasets, device=device,
                              pretrain_args=pretrain_args)
    model.train()
    pretrain_args.to_json()
    wandb.finish()
    cprint(f"🌟 TabSTAR was pretrained. The experiment name is: {pretrain_args.full_exp_name}")
