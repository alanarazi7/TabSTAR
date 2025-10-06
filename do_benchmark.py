import argparse

from tabstar.tabstar_model import BaseTabSTAR
from tabstar.training.devices import get_device
from tabstar_paper.baselines.carte import CARTE
from tabstar_paper.baselines.catboost import CatBoost, CatBoostOpt
from tabstar_paper.baselines.lgbm import LightGBM, LightGBMOpt
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.realmlp import RealMLP, RealMLPOpt
from tabstar_paper.baselines.tabdpt import TabDPT
from tabstar_paper.baselines.tabicl import TabICL
from tabstar_paper.baselines.tabm import TabM, TabMOpt
from tabstar_paper.baselines.tabpfnv2 import TabPFNv2
from tabstar_paper.baselines.xgboost import XGBoost, XGBoostOpt
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.constants import DEVICE
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.logging import wandb_run, wandb_finish

BASELINES = [CatBoost, CatBoostOpt, XGBoost, XGBoostOpt, LightGBM, LightGBMOpt, RandomForest,
             RealMLP, RealMLPOpt, TabM, TabMOpt,
             CARTE,
             TabICL, TabDPT, TabPFNv2]

baseline_names = {model.SHORT_NAME: model for model in BASELINES}
SHORT2MODELS = {'tabstar': BaseTabSTAR, **baseline_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--carte_lr_idx', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    model = SHORT2MODELS[args.model]
    dataset = get_dataset_from_arg(args.dataset_id)
    device = get_device(device=DEVICE)

    if dataset.name.startswith("REG_") and args.model == "icl":
        print(f"Skipping {dataset.name} for TabICL as it is a regression dataset.")
        exit()
    idx2str = {None: "", 0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
    exp_name = f"{args.model}_{dataset.name}_{args.fold}{idx2str[args.carte_lr_idx]}"
    wandb_run(exp_name=exp_name, project="tabstar_benchmark")
    ret = evaluate_on_dataset(
        model_cls=model,
        dataset_id=dataset,
        fold=args.fold,
        train_examples=args.train_examples,
        device=device,
        verbose=args.verbose,
        carte_lr_idx=args.carte_lr_idx,
    )
    wandb_finish(d_summary=ret)
