import argparse
import logging
import os

import torch
from tqdm import tqdm

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.tabstar_model import BaseTabSTAR
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset, DOWNSTREAM_EXAMPLES
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS
from tabstar_paper.datasets.downloading import get_dataset_from_arg
from tabstar_paper.utils import log_calls
from tabular.utils.io_handlers import dump_json

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(SHORT2MODELS.keys()), default=None)
    parser.add_argument('--dataset_id', default=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION.value)
    parser.add_argument('--run_num', type=int)
    parser.add_argument('--train_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    args = parser.parse_args()

    models = BASELINES
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

    for model, dataset_id, run_num in tqdm(combos):
        model_name = model.__class__.__name__
        key_file = f".benchmark_results/{model_name}_{dataset_id.name}_{run_num}.txt"
        if os.path.exists(key_file):
            continue
        metric = evaluate_on_dataset(model_cls=model, dataset_id=dataset_id, run_num=run_num,
                                     train_examples=args.train_examples, device=device)
        result = {"metric": metric, "dataset": dataset_id.name, "run_num": run_num, 'model': model_name}
        dump_json(result, key_file)


if __name__ == "__main__":
    main()