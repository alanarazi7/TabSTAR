import argparse
import os
import time

import torch
from pandas import DataFrame
from tqdm import tqdm

from tabstar.tabstar_model import BaseTabSTAR
from tabstar.training.devices import get_device
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.random_forest import RandomForest
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS
from tabstar_paper.constants import GPU
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.io_handlers import dump_json, load_json_lines
from tabstar_paper.utils.logging import get_current_commit_hash

BASELINES = [CatBoost, XGBoost, RandomForest]

baseline_names = {model.SHORT_NAME: model for model in BASELINES}
SHORT2MODELS = {'tabstar': BaseTabSTAR, **baseline_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id')
    parser.add_argument('--trial', type=int)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--cls', action='store_true', default=False)
    args = parser.parse_args()

    models = list(SHORT2MODELS.values())
    trials = list(range(10))
    datasets = TEXTUAL_DATASETS

    if args.model:
        models = [SHORT2MODELS[args.model]]
    if args.dataset_id:
        datasets = [get_dataset_from_arg(args.dataset_id)]
    if isinstance(args.trial, int):
        trials = [args.trial]

    device = get_device(device=GPU)
    combos = [(m, d, r) for m in models for d in datasets for r in trials]
    if device is not None:
        count_gpus_in_machine = torch.cuda.device_count()
        combos = combos[int(GPU)::count_gpus_in_machine]

    existing = DataFrame(load_json_lines("tabstar_paper/benchmarks/benchmark_runs.txt"))
    existing_combos = {(d['model'], d['dataset'], d.get('run_num') or d.get('trial')) for _, d in existing.iterrows()}
    for model, dataset_id, trial in tqdm(combos):
        if args.cls and dataset_id.name.startswith("REG_"):
            continue
        model_name = model.__name__
        if (model_name, dataset_id.name, trial) in existing_combos:
            continue
        key_file = f".benchmark_results/{model_name}_{dataset_id.name}_{trial}.txt"
        if os.path.exists(key_file):
            continue
        print(f"Evaluating {model_name} on {dataset_id.name} with trial {trial}")
        start_time = time.time()
        metrics = evaluate_on_dataset(
            model_cls=model,
            dataset_id=dataset_id,
            trial=trial,
            train_examples=args.train_examples,
            device=device
        )
        result = {
            "score": metrics.score,
            "dataset": dataset_id.name,
            "trial": trial,
            "model": model_name,
            "metrics": dict(metrics.metrics),
            "runtime": time.time() - start_time,
            "device": device,
            "train_examples": args.train_examples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "git":  get_current_commit_hash(),
        }
        key_file_dir = os.path.dirname(key_file)
        os.makedirs(key_file_dir, exist_ok=True)
        dump_json(result, key_file)