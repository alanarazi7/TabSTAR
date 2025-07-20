import argparse
from dataclasses import asdict
from os.path import exists
from typing import List

import wandb

from tabstar.datasets.all_datasets import OpenMLDatasetID, TabularDatasetID
from tabstar.datasets.benchmark_folds import TEXT2FOLD
from tabstar.datasets.pretrain_folds import PRETRAIN2FOLD
from tabstar.training.devices import get_device
from tabstar.training.hyperparams import MAX_EPOCHS
from tabstar_paper.constants import GPU
from tabstar_paper.pretraining.hyperparameters import TABULAR_LAYERS, TEXTUAL_UNFREEZE_LAYERS, BASE_LR, WEIGHT_DECAY
from tabstar_paper.pretraining.pretrainer import TabSTARPretrainer
from tabstar_paper.utils.logging import wandb_run
from tabular.benchmarks.all_datasets import ANALYSIS_TEXT_DOWNSTREAM
from tabular.trainers.pretrain_args import PretrainArgs


def do_pretrain(pretrain_datasets: List[TabularDatasetID],
                pretrain_args: PretrainArgs):
    if exists(pretrain_args.path):
        print(f"Pretraining model already exists for {pretrain_args.full_exp_name}")
        return
    print(f"🧪 Initializing experiment {pretrain_args.full_exp_name}")
    device = get_device(device=GPU)
    wandb_run(exp_name=pretrain_args.raw_exp_name, project="tabstar_pretrain")
    wandb.config.update(asdict(pretrain_args), allow_val_change=True)
    print(f"Pretraining over {len(pretrain_datasets)} datasets")
    model = TabSTARPretrainer(run_name=pretrain_args.full_exp_name,
                              dataset_ids=pretrain_datasets,
                              max_epochs=pretrain_args.epochs,
                              device=device,
                              pretrain_args=pretrain_args)
    model.train()
    pretrain_args.to_json()
    print(f"🌟 TabSTAR was pretrained. The experiment name is: {pretrain_args.full_exp_name}")
    wandb.finish()


def define_downstream_datasets(arg: argparse.Namespace) -> List[TabularDatasetID]:
    if arg.analysis:
        return ANALYSIS_TEXT_DOWNSTREAM
    if arg.fold is None:
        return []
    fold_dict = TEXT2FOLD if args.only_text_folds else PRETRAIN2FOLD
    datasets = [d for d, f in fold_dict.items() if f == arg.fold]
    return datasets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the training script with optional arguments.")
    # General
    parser.add_argument('--exp', type=str, default="default_pretrain_exp")
    parser.add_argument('--analysis', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    # Arch
    parser.add_argument('--tabular_layers', type=int, default=TABULAR_LAYERS)
    parser.add_argument('--e5_unfreeze_layers', type=int, default=TEXTUAL_UNFREEZE_LAYERS)
    # Data
    parser.add_argument('--n_datasets', type=int, default=None)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--only_text_folds', action='store_true', default=False)
    # Optimizer
    parser.add_argument('--base_lr', type=float, default=BASE_LR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)

    args = parser.parse_args()

    downstream_data = define_downstream_datasets(args)

    pretrain_data = [d for d in PRETRAIN2FOLD if d not in downstream_data]

    if args.n_datasets is not None:
        pretrain_data = pretrain_data[:args.n_datasets]
    elif args.debug:
        pretrain_data = [OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
                         OpenMLDatasetID.MUL_NATURE_EUCALYPTUS_SEED,
                         OpenMLDatasetID.REG_SPORTS_MONEYBALL]
        args.epochs = 1

    # TODO: use HfArgumentParser probably
    pretraining_args = PretrainArgs.from_args(args=args, pretrain_data=pretrain_data)

    do_pretrain(pretrain_datasets=pretrain_data, pretrain_args=pretraining_args)