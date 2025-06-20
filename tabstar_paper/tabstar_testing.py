import torch

from tabstar_paper.do_benchmark import eval_tabstar_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS
from tabular.utils.io_handlers import dump_json

import os

datasets = sorted(TEXTUAL_DATASETS, key=lambda d: d.name)
runs = [(dataset, run_num) for dataset in datasets for run_num in range(10)]
GPU = os.getenv("GPU", None)
device = GPU
if device is not None:
    device = "cuda:{GPU}"
    count_gpus_in_machine = torch.cuda.device_count()
    runs = runs[GPU::count_gpus_in_machine]


for dataset, run_num in runs:
    key_file = f"benchmark_results_v6/tabstar_{dataset.name}_{run_num}.txt"
    if os.path.exists(key_file):
        continue
    print(f"Evaluating TabSTAR on {dataset.name} run {run_num}...")
    metric = eval_tabstar_on_dataset(dataset_id=dataset, run_num=run_num, train_examples=10_000, device=gpu)
    result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
    dump_json(result, key_file)