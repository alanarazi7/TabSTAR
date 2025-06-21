from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.do_benchmark import eval_tabstar_on_dataset


run_num= 0
dataset = OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING
metric = eval_tabstar_on_dataset(dataset_id=dataset, run_num=run_num, train_examples=10_000)
print(f"Scored {metric:.4f} on dataset {dataset.name} run {run_num}.")