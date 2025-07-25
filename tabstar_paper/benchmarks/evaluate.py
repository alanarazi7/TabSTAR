from typing import Type

import torch

from tabstar.constants import SEED
from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARClassifier, BaseTabSTAR, TabSTARRegressor
from tabstar.training.metrics import Metrics
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.preprocessing.sampling import subsample_dataset

DOWNSTREAM_EXAMPLES = 10_000
TRIALS = 10


def evaluate_on_dataset(model_cls: Type[TabularModel],
                        dataset_id: TabularDatasetID,
                        trial: int,
                        train_examples: int,
                        device: torch.device,
                        verbose: bool = False) -> Metrics:
    is_tabstar = issubclass(model_cls, BaseTabSTAR)
    name = "TabSTAR ⭐" if is_tabstar else model_cls.MODEL_NAME
    print(f"Running model {name} over dataset {dataset_id} with trial {trial}")
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, trial=trial)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, trial=trial, train_examples=train_examples)
    if is_tabstar:
        tabstar_cls = TabSTARClassifier if is_cls else TabSTARRegressor
        model = tabstar_cls(pretrain_dataset_or_path=dataset_id, device=device, verbose=verbose, random_state=SEED)
    else:
        model = model_cls(is_cls=is_cls, device=device, verbose=verbose)
    model.fit(x_train, y_train)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    return metrics