import argparse

from tabular.benchmarks.all_datasets_shuffled import ALL_SHUFFLED_DATASETS
from tabular.benchmarks.all_datasets import ANALYSIS_TEXT_DOWNSTREAM
from tabular.benchmarks.cross_validation import get_downstream_fold
from tabular.tabstar.params.constants import (TEXTUAL_UNFREEZE_LAYERS, BASE_LR, WEIGHT_DECAY,
                                              NumberVerbalization, TABULAR_LAYERS)
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.pretraining import do_pretrain

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the training script with optional arguments.")
    # General
    parser.add_argument('--exp', type=str, default="default_pretrain_exp")
    parser.add_argument('--production', action='store_true', default=False)
    # Arch
    parser.add_argument('--tabular_layers', type=int, default=TABULAR_LAYERS)
    parser.add_argument('--e5_unfreeze_layers', type=int, default=TEXTUAL_UNFREEZE_LAYERS)
    # Data
    parser.add_argument('--n_datasets', type=int, default=None)
    parser.add_argument('--numbers_verbalization', default="full", choices=[v.value for v in NumberVerbalization])
    parser.add_argument('--fold', type=int, default=None)
    # Optimizer
    parser.add_argument('--base_lr', type=float, default=BASE_LR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)

    args = parser.parse_args()

    if args.fold is not None:
        downstream_data = get_downstream_fold(k=args.fold, folds=args.k_folds)
    elif args.production:
        downstream_data = []
    else:
        downstream_data = ANALYSIS_TEXT_DOWNSTREAM

    pretrain_data = [d for d in ALL_SHUFFLED_DATASETS if d not in downstream_data]

    if args.n_datasets is not None:
        pretrain_data = pretrain_data[:args.n_datasets]

    # TODO: use HfArgumentParser probably
    pretrain_args = PretrainArgs.from_args(args=args, pretrain_data=pretrain_data)

    do_pretrain(pretrain_datasets=pretrain_data,
                downstream_datasets=downstream_data,
                pretrain_args=pretrain_args)