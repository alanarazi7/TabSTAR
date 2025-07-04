import argparse
import logging
import os
import time

import torch
from pandas import DataFrame
from tqdm import tqdm

from tabstar.tabstar_model import BaseTabSTAR
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils.io_handlers import dump_json, load_json_lines
from tabstar_paper.utils.logging import log_calls, get_current_commit_hash

BASELINES = [CatBoost, XGBoost]

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
    datasets = [
                313, # spectrometer
                # 372, # internet_usage
                # 390, # new3s.wc
                # 1457, # amazon-commerce-reviews
                1472, # energy-efficiency
                1482, # leaf
                # 1491, # one-hundred-plants-margin
                # 1492, 1493,
                4552, # BachChoralHarmony
                # 40923, # Devnagari-Script
                # 40971, # collins
                # 41039, # EMNIST_Balanced
                # 41083, # Olivetti_Faces
                # 41167, # dionis
                # 41169, # helena
                41960, # seattlecrime6
                # 41983,
                # 41986, 41988, 41989, 41990, 41991, 42078, 42087, 42088,
                # 42089, 42123, 42133, 42166, 42223, 42396,
                43723, # Toronto-Apartment-Rental-Price
                # 44281, 44282, 44283, 44284, 44285, 44286, 44288,
                # 44289, 44290, 44291, 44292, 44294, 44298, 44300, 44304,
                # 44305, 44306, 44307, 44316, 44317, 44318, 44319, 44320,
                # 44321, 44322, 44323, 44324, 44325, 44326, 44328, 44331,
                # 44333, 44337, 44338, 44340, 44341, 44478, 44479, 44480,
                # 44481, 44482, 44533, 44534, 44535, 44536, 44537, 44728,
                # 44729, 44730, 44731, 44732, 45049, 45102, 45103, 45104,
                # 45274, 45569,
                45923, # IndoorScenes
                # 45936, 46577, 46578,
                # 46608, # drug_reviews_druglib_com
                # 46649,
                # 46653, 46678, 46686, 46702, 46770, 46782, 46804, 46813,
                # 46816, 46852, 46887
                ]


    if args.model:
        models = [SHORT2MODELS[args.model]]
    # if args.dataset_id:
    #     datasets = [get_dataset_from_arg(args.dataset_id)]
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
    existing = DataFrame(load_json_lines("tabstar_paper/benchmarks/benchmark_runs.txt"))
    existing_combos = {(d['model'], d['dataset'], d['run_num']) for _, d in existing.iterrows()}
    for model, dataset_id, run_num in tqdm(combinations):
        # if args.cls and dataset_id.name.startswith("REG_"):
        #     continue
        model_name = model.__name__
        if (model_name, dataset_id, run_num) in existing_combos:
            continue
        key_file = f".benchmark_results/{model_name}_{dataset_id}_{run_num}.txt"
        if os.path.exists(key_file):
            continue
        print(f"Evaluating {model_name} on {dataset_id} with run num {run_num}")
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
            "dataset": dataset_id,
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