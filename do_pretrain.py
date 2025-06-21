import argparse

from tabstar.datasets.benchmark_folds import TEXT2FOLD
from tabstar.datasets.pretrain_folds import PRETRAIN2FOLD
from tabstar_paper.pretraining.hyperparameters import TABULAR_LAYERS, TEXTUAL_UNFREEZE_LAYERS, BASE_LR, WEIGHT_DECAY
from tabular.benchmarks.all_datasets_shuffled import ALL_SHUFFLED_DATASETS
from tabular.benchmarks.all_datasets import ANALYSIS_TEXT_DOWNSTREAM
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.pretraining import do_pretrain


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the training script with optional arguments.")
    # General
    parser.add_argument('--exp', type=str, default="default_pretrain_exp")
    parser.add_argument('--analysis', action='store_true', default=False)
    # Arch
    parser.add_argument('--tabular_layers', type=int, default=TABULAR_LAYERS)
    parser.add_argument('--e5_unfreeze_layers', type=int, default=TEXTUAL_UNFREEZE_LAYERS)
    # Data
    parser.add_argument('--n_datasets', type=int, default=None)
    parser.add_argument('--numbers_verbalization', default="full", choices=[v.value for v in NumberVerbalization])
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--only_text_folds', action='store_true', default=False)
    # Optimizer
    parser.add_argument('--base_lr', type=float, default=BASE_LR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)

    args = parser.parse_args()

    if args.fold is not None:
        fold_dict = TEXT2FOLD if args.only_text_folds else PRETRAIN2FOLD
        downstream_data = fold_dict[args.fold]
    elif args.analysis:
        downstream_data = ANALYSIS_TEXT_DOWNSTREAM
    else:
        downstream_data = []

    pretrain_data = [d for d in ALL_SHUFFLED_DATASETS if d not in downstream_data]

    if args.n_datasets is not None:
        pretrain_data = pretrain_data[:args.n_datasets]

    # TODO: use HfArgumentParser probably
    pretrain_args = PretrainArgs.from_args(args=args, pretrain_data=pretrain_data)

    do_pretrain(pretrain_datasets=pretrain_data,
                downstream_datasets=downstream_data,
                pretrain_args=pretrain_args)