from typing import List

from tabular.benchmarks.utils import get_suite_datasets
from tabular.datasets.tabular_datasets import OpenMLDatasetID

# Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning
# https://arxiv.org/abs/2007.04074

# https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=293
def get_amlb_train_datasets() -> List[OpenMLDatasetID]:
    datasets = get_suite_datasets(sid=293, name='AutoML-Benchmark-Train', n_datasets=208)
    openml_datasets = [OpenMLDatasetID(d) for d in datasets]
    return openml_datasets
