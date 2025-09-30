from dataclasses import dataclass
from typing import Type

import pandas as pd
from pandas import DataFrame

from tabpfn_augmentor.evaluator import evaluate_on_augmented_dataset
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.training.devices import get_device
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.realmlp import RealMLP
from tabstar_paper.baselines.tabpfnv2 import TabPFNv2
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.leaderboard.data.keys import DATASET, MODEL, TEST_SCORE

augment_benchmark = [
    OpenMLDatasetID.BIN_ANONYM_ALBERT,
    OpenMLDatasetID.REG_ANONYM_HOUSE_16H,
    # OpenMLDatasetID.REG_HOUSES_CALIFORNIA_PRICES_2020,
    # OpenMLDatasetID.MUL_NATURE_YEAST_PROTEIN,
    # OpenMLDatasetID.BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN,
    # OpenMLDatasetID.BIN_COMPUTERS_IMAGE_BANK_NOTE_AUTHENTICATION,
]

models = [XGBoost, TabPFNv2] #RealMLP, TabPFNv2]

num_folds = 3

augmentations = [True, False]

max_examples = 1000
max_features = 8

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
                                            do_augment=t.do_augment, fold=t.fold, device=device,
                                            train_examples=max_examples, max_features=max_features)
        results.append(res)
    df = DataFrame(results)
    pivot_df = df.pivot_table(index=[DATASET], columns=[MODEL, "augment"], values=[TEST_SCORE])
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)
    print(pivot_df)


if __name__ == "__main__":
    eval_augmented_benchmark()
