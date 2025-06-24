import logging
import os

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.do_baseline import eval_baseline_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS
from tabular.utils.io_handlers import dump_json


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s') 


datasets = sorted(TEXTUAL_DATASETS, key=lambda d: d.name)
datasets = [d for d in datasets if not d.name.startswith('REG_')]

# this scripts runs on all datasets (TEXTUAL_DATASETS) that were used for the benchmarking in the TabSTAR paper.
# it runs 10 times on each dataset, currently only for XGBoost.
# it creates a json 

for dataset in datasets:
    print(f"Running benchmarks for dataset: {dataset.name}")
    for run_num in range(10):
        print(f"Run number: {run_num}")
        for model in [XGBoost]:
            key_file = f"benchmark_result/{model.SHORT_NAME}_{dataset.name}_{run_num}.txt"
            if not os.path.exists(key_file):
                print(f"Evaluating {model.SHORT_NAME} on {dataset.name} run {run_num}...")
                metric = eval_baseline_on_dataset(model=model, dataset_id=dataset, run_num=run_num, train_examples=10_000)
                result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
                dump_json(result, key_file)