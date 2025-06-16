import os

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.do_baseline import eval_baseline_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS, TEXTUAL_BIG
from tabular.utils.io_handlers import dump_json

for dataset in TEXTUAL_DATASETS:
    if dataset.name.startswith('REG_'):
        # Skip regression datasets for now, we only evaluate classification
        continue
    if dataset in TEXTUAL_BIG:
        # Skip for now big datasets, we need to handle the num_examples
        continue
    for run_num in range(10):
        key_file = f"benchmark_catboost_results/{dataset.name}_{run_num}.txt"
        if os.path.exists(key_file):
            continue
        print(f"Evaluating {dataset.name} run {run_num}...")
        metric = eval_baseline_on_dataset(model=CatBoost, dataset_id=dataset, run_num=run_num, train_examples=10_000)
        result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
        dump_json(result, key_file)