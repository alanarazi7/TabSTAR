import argparse

from tabstar.tabstar_model import BaseTabSTAR
from tabstar.training.devices import get_device
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.tabdpt import TabDPT
from tabstar_paper.baselines.tabicl import TabICL
from tabstar_paper.baselines.tabpfn import TabPFN
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.constants import DEVICE
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.logging import wandb_run, wandb_finish

BASELINES = [CatBoost, XGBoost, RandomForest,
             RealMLP,
             TabICL, TabDPT, TabPFN]

baseline_names = {model.SHORT_NAME: model for model in BASELINES}
SHORT2MODELS = {'tabstar': BaseTabSTAR, **baseline_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    model = SHORT2MODELS[args.model]
    dataset = get_dataset_from_arg(args.dataset_id)
    device = get_device(device=DEVICE)

    if dataset.name.startswith("REG_") and args.model == "icl":
        print(f"Skipping {dataset.name} for TabICL as it is a regression dataset.")
        exit()
    wandb_run(exp_name=f"{args.model}_{dataset.name}_{args.fold}", project="tabstar_benchmark")
    ret = evaluate_on_dataset(
        model_cls=model,
        dataset_id=dataset,
        fold=args.fold,
        train_examples=args.train_examples,
        device=device,
        verbose=args.verbose
    )
    wandb_finish(d_summary=ret)
