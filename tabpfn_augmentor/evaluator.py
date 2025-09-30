import time
from dataclasses import asdict
from typing import Type, Dict, Optional

import pandas as pd
import torch

from tabpfn_augmentor.generate_data import augment_with_tabpfn
from tabpfn_augmentor.data_cleaning import remove_semantics, fill_nulls
from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.datasets.downloading import download_dataset
from tabstar_paper.datasets.objects import SupervisedTask
from tabstar_paper.preprocessing.sampling import subsample_dataset


def evaluate_on_augmented_dataset(model_cls: Type[TabularModel],
                                  dataset_id: TabularDatasetID,
                                  do_augment: bool,
                                  fold: int,
                                  device: torch.device,
                                  train_examples: int,
                                  max_features: int) -> Dict:
    start_time = time.time()
    name = model_cls.MODEL_NAME
    print(f"Running model {name} over dataset {dataset_id} with fold {fold}")
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, fold=fold)
    x, y = remove_semantics(x, y)
    x, y = fill_nulls(x, y)
    x = x.iloc[:, :max_features]
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=fold, train_examples=train_examples)
    if do_augment:
        cache_key = f"{dataset_id.name}_fold{fold}_train{train_examples}_feat{max_features}"
        synth_x, synth_y = augment_with_tabpfn(x_train, y_train, is_cls=is_cls, cache_key=cache_key)
        x_train = pd.concat([x_train, synth_x], ignore_index=True)
        y_train = pd.concat([y_train, synth_y], ignore_index=True)
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
        "max_features": max_features,
           }
    print(f"Scored {metrics.score:.4f} on dataset {dataset_id.name}, fold {fold} in {int(runtime)} seconds,"
          f"augment: {do_augment}, train_samples {train_examples}, max features {max_features}.")
    return d_summary