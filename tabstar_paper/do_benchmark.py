import torch

from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.do_baseline import eval_baseline_on_dataset
from tabstar_paper.do_tabstar import eval_tabstar_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS
from tabular.utils.io_handlers import dump_json

import os

datasets = sorted(TEXTUAL_DATASETS, key=lambda d: d.name)
datasets = [d for d in datasets if not d.name.startswith('REG_')]
runs = [(dataset, run_num) for dataset in datasets for run_num in range(10)]
GPU = os.getenv("GPU", None)
device = GPU
if device is not None:
    device = f"cuda:{GPU}"
    count_gpus_in_machine = torch.cuda.device_count()
    runs = runs[int(GPU)::count_gpus_in_machine]


for dataset, run_num in runs:
    key_file = f".benchmark_results/tabstar_{dataset.name}_{run_num}.txt"
    if os.path.exists(key_file):
        continue
    print(f"Evaluating TabSTAR on {dataset.name} run {run_num}...")
    metric = eval_tabstar_on_dataset(dataset_id=dataset, run_num=run_num, train_examples=10_000, device=device)
    result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
    dump_json(result, key_file)
    # for model in [CatBoost]:
    #     key_file = f".benchmark_results/{model.SHORT_NAME}_{dataset.name}_{run_num}.txt"
    #     if not os.path.exists(key_file):
    #         print(f"Evaluating {model.SHORT_NAME} on {dataset.name} run {run_num}...")
    #         metric = eval_baseline_on_dataset(model=model, dataset_id=dataset, run_num=run_num, train_examples=10_000)
    #         result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
    #         dump_json(result, key_file)
