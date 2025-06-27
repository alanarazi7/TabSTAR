import argparse
import logging
import os
import time

import torch
from tqdm import tqdm

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.tabstar_model import BaseTabSTAR
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.io_handlers import dump_json
from tabstar_paper.utils.logging import log_calls, get_current_commit_hash

BASELINES = [CatBoost] #, XGBoost]

baseline_names = {model.SHORT_NAME: model for model in BASELINES}
SHORT2MODELS = {'tabstar': BaseTabSTAR, **baseline_names}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s') # as a default, will only print warnings and errors. \
# locally, you can set it to DEBUG or INFO to see more details.

GPU = os.getenv("GPU", None)
device = GPU
if device is not None:
    device = f"cuda:{GPU}"


@log_calls
def main():
    """
    Entry point for running benchmarks on tabular models.
    Parses arguments, prepares model/dataset/run combinations, and evaluates each.
    """
    args = parse_args()
    combinations = prepare_combinations(args)
    run_benchmarks(combinations, args)


@log_calls
def parse_args():
    """Parse command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id')
    parser.add_argument('--run_num', type=int)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--cls', action='store_true', default=False)
    return parser.parse_args()


@log_calls
def prepare_combinations(args):
    """
    Prepare all (model, dataset, run_num) combinations to evaluate.
    Returns a list of tuples.
    """
    models = list(SHORT2MODELS.values())
    run_numbers = list(range(10))
    datasets = TEXTUAL_DATASETS

    if args.model:
        models = [SHORT2MODELS[args.model]]
    if args.dataset_id:
        datasets = [get_dataset_from_arg(args.dataset_id)]
    if isinstance(args.run_num, int):
        run_numbers = [args.run_num]

    combos = [(m, d, r) for m in models for d in datasets for r in run_numbers]

    if device is not None:
        count_gpus_in_machine = torch.cuda.device_count()
        combos = combos[int(GPU)::count_gpus_in_machine]

    return combos


@log_calls
def run_benchmarks(combinations, args):
    """
    Run evaluation for each (model, dataset, run_num) combination.
    Saves results to local files.
    """
    for model, dataset_id, run_num in tqdm(combinations):
        if args.cls and dataset_id.name.startswith("REG_"):
            continue
        model_name = model.__name__
        key_file = f".benchmark_results/{model_name}_{dataset_id.name}_{run_num}.txt"
        print(f"Evaluating {model_name} on {dataset_id.name} with run num {run_num}")
        if os.path.exists(key_file):
            continue
        start_time = time.time()
        metrics = evaluate_on_dataset(
            model_cls=model,
            dataset_id=dataset_id,
            run_num=run_num,
            train_examples=args.train_examples,
            device=device
        )
        result = {
            "score": metrics.score,
            "dataset": dataset_id.name,
            "run_num": run_num,
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


if __name__ == "__main__":
    main()