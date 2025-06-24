from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset
from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS

# TODO: these imports shouldn't exist, they are from the tabular repo
from tabular.utils.io_handlers import dump_json

import os

runs = [(dataset, run_num) for dataset in TEXTUAL_DATASETS for run_num in range(10)]

for dataset, run_num in runs:
    for model in [CatBoost]:
        key_file = f".benchmark_results/{model.SHORT_NAME}_{dataset.name}_{run_num}.txt"
        if not os.path.exists(key_file):
            print(f"Evaluating {model.SHORT_NAME} on {dataset.name} run {run_num}...")
            metric = evaluate_on_dataset(model=model, dataset_id=dataset, run_num=run_num)
            result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
            dump_json(result, key_file)
