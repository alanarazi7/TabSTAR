import argparse

import torch

from tabular.datasets.tabular_datasets import get_dataset_from_arg
from tabular.evaluation.constants import DOWNSTREAM_EXAMPLES, N_RUNS
from tabular.models.competitors.carte import CARTE
from tabular.models.competitors.catboost import CatBoost, CatBoostOptuna
from tabular.models.competitors.random_forest import RandomForest
from tabular.models.competitors.tabpfn2 import TabPFNv2
from tabular.models.competitors.xg_boost import XGBoost, XGBoostOptuna
from tabular.trainers.finetune import do_finetune_run
from tabular.utils.gpus import get_device
from tabular.utils.utils import cprint

BASELINES = [CatBoost, TabPFNv2, CARTE, RandomForest, XGBoost, CatBoostOptuna, XGBoostOptuna]

SHORT2MODELS = {model.SHORT_NAME: model for model in BASELINES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default_baseline_experiment')
    parser.add_argument('--model', type=str, default='cat', choices=SHORT2MODELS.keys())
    parser.add_argument('--dataset_id', default=46667)
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--n_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--carte_lr_index', type=int, default=None)
    args = parser.parse_args()

    cprint(f"🧹 Running {args.exp} with {args.model} on dataset {args.dataset_id} for run {args.run_num}")

    model = SHORT2MODELS[args.model]
    if model == CARTE and not 0 <= args.carte_lr_index <= 5:
        raise ValueError(f"Invalid CARTE lr index: {args.carte_lr_index}. Should be between 0 and 5.")

    dataset = get_dataset_from_arg(args.dataset_id)
    device = torch.device(get_device())

    assert 0 <= args.run_num < N_RUNS, f"Invalid run number: {args.run_num}. Should be between 0 and {N_RUNS - 1}"
    n_examples = args.n_examples

    do_finetune_run(exp_name=args.exp, dataset=dataset, model=model, run_num=args.run_num, train_examples=n_examples,
                    device=device)