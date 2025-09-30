from dataclasses import dataclass
from typing import Type

from tabpfn_augmentor.evaluator import evaluate_on_augmented_dataset
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.training.devices import get_device
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.tabpfnv2 import TabPFNv2
from tabstar_paper.baselines.xgboost import XGBoost

augment_benchmark = [
    OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN,
    OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN,
    OpenMLDatasetID.REG_SPORTS_MONEYBALL,
    OpenMLDatasetID.MUL_NATURE_EUCALYPTUS_SEED,
]

models = [XGBoost, RealMLP, TabPFNv2]

num_folds = 5

augmentations = [True, False]

@dataclass
class TabularTask:
    dataset: OpenMLDatasetID
    model: Type[TabularModel]
    fold: int
    do_augment: bool


tabular_tasks = [TabularTask(dataset=d, model=m, fold=f, do_augment=a)
                 for d in augment_benchmark
                 for m in models
                 for f in range(num_folds)
                 for a in augmentations]


def eval_augmented_benchmark():
    results = []
    device = get_device()
    for t in tabular_tasks:
        res = evaluate_on_augmented_dataset(model_cls=t.model, dataset_id=t.dataset,
                                            do_augment=t.do_augment, fold=t.fold, device=device)
        results.append(res)


if __name__ == "__main__":
    eval_augmented_benchmark()
