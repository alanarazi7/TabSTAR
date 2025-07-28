import argparse

from tabstar.tabstar_model import BaseTabSTAR
from tabstar.training.devices import get_device
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.tabdpt import TabDPT
from tabstar_paper.baselines.tabicl import TabICL
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.constants import DEVICE
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.logging import wandb_run, wandb_finish

BASELINES = [CatBoost, XGBoost, RandomForest, TabICL, TabDPT]

baseline_names = {model.SHORT_NAME: model for model in BASELINES}
SHORT2MODELS = {'tabstar': BaseTabSTAR, **baseline_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    args = parser.parse_args()

    model = SHORT2MODELS[args.model]
    dataset = get_dataset_from_arg(args.dataset_id)
    if issubclass(model, TabularModel) and not model.ALLOW_GPU:
        DEVICE = "cpu"
    device = get_device(device=DEVICE)

    if dataset.name.startswith("REG_") and args.model == "icl":
        print(f"Skipping {dataset.name} for TabICL as it is a regression dataset.")
        exit()
    wandb_run(exp_name=f"{model.SHORT_NAME}_{dataset.name}_{args.trial}", project="tabstar_baseline")
    ret = evaluate_on_dataset(
        model_cls=model,
        dataset_id=dataset,
        trial=args.trial,
        train_examples=args.train_examples,
        device=device
    )
    wandb_finish(d_summary=ret)
