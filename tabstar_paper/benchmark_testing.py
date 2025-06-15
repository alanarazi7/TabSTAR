import os
from random import shuffle

from tabstar_paper.do_benchmark import eval_tabstar_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS, TEXTUAL_BIG
from tabular.utils.io_handlers import dump_json

for dataset in sorted(set(TEXTUAL_DATASETS).difference(TEXTUAL_BIG)):
    for run_num in range(10):
        key_file = f"benchmark_results/{dataset.name}_{run_num}.txt"
        if os.path.exists(key_file):
            continue
        print(f"Evaluating {dataset.name} run {run_num}...")
        metric = eval_tabstar_on_dataset(dataset_id=dataset, run_num=run_num, train_examples=10_000)
        result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
        dump_json(result, key_file)