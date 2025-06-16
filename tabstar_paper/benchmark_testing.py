import os

from tabstar_paper.do_baseline import BASELINES, eval_baseline_on_dataset
from tabstar_paper.do_benchmark import eval_tabstar_on_dataset
from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS, TEXTUAL_BIG
from tabular.utils.io_handlers import dump_json

datasets = sorted(TEXTUAL_DATASETS, key=lambda d: d.name)

for dataset in datasets:
    if dataset.name.startswith('REG_'):
        # Skip regression datasets for now, we only evaluate classification
        continue
    if dataset in TEXTUAL_BIG:
        # Skip for now big datasets, we need to handle the num_examples
        continue
    for run_num in range(10):
        # key_file = f"benchmark_results/tabstar_{dataset.name}_{run_num}.txt"
        # if not os.path.exists(key_file):
        #     print(f"Evaluating {dataset.name} run {run_num}...")
        #     metric = eval_tabstar_on_dataset(dataset_id=dataset, run_num=run_num, train_examples=10_000)
        #     result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
        #     dump_json(result, key_file)
        for model in BASELINES:
            key_file = f"benchmark_results/{model.SHORT_NAME}_{dataset.name}_{run_num}.txt"
            if not os.path.exists(key_file):
                print(f"Evaluating {model.SHORT_NAME} on {dataset.name} run {run_num}...")
                metric = eval_baseline_on_dataset(model=model, dataset_id=dataset, run_num=run_num, train_examples=10_000)
                result = {"metric": metric, "dataset": dataset.name, "run_num": run_num}
                dump_json(result, key_file)