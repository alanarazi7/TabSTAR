import time
from dataclasses import asdict
from typing import Type, Dict

import torch

from tabpfn_augmentor.generate_data import augment_with_tabpfn
from tabpfn_augmentor.numericalization import remove_semantics
from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.benchmarks.evaluate import DOWNSTREAM_EXAMPLES
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.datasets.objects import SupervisedTask
from tabstar_paper.preprocessing.sampling import subsample_dataset


def evaluate_on_augmented_dataset(model_cls: Type[TabularModel],
                                  dataset_id: TabularDatasetID,
                                  do_augment: bool,
                                  fold: int,
                                  device: torch.device,
                                  train_examples: int = DOWNSTREAM_EXAMPLES) -> Dict:
    start_time = time.time()
    name = model_cls.MODEL_NAME
    print(f"Running model {name} over dataset {dataset_id} with fold {fold}")
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, fold=fold)
    x, y = remove_semantics(x, y)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=fold, train_examples=train_examples)
    if do_augment:
         # TODO: for efficiency and fairness, the augmentation should happen only once per dataset/fold for all methods
         exp_synthetic = augment_with_tabpfn(x_train, y_train, is_cls=is_cls)
         raise NotImplementedError("Augmentation not implemented yet.")
    prefix2task = {"REG": SupervisedTask.REGRESSION, "BIN": SupervisedTask.BINARY, "MUL": SupervisedTask.MULTICLASS}
    problem_type = prefix2task[dataset_id.name[:3]]
    model = model_cls(problem_type=problem_type, device=device)
    model.fit(x_train, y_train)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    runtime = time.time() - start_time
    d_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model": name,
        "dataset": dataset_id.name,
        "fold": fold,
        "train_examples": train_examples,
        "test_score": metrics.score,
        "metrics_dict": asdict(metrics),
        "runtime": runtime,
           }
    print(f"Scored {metrics.score:.4f} on dataset {dataset_id.name}, fold {fold} in {int(runtime)} seconds.")
    return d_summary