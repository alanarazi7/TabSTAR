from typing import Type

import pytest
import torch

from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset


def _test_evaluate(model_cls: Type[TabularModel], fold: int = 0) -> float:
    d_summary = evaluate_on_dataset(model_cls=model_cls, dataset_id=OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN,
                                    fold=fold, device=torch.device("cpu"), train_examples=100, verbose=False)
    return d_summary["test_score"]




def test_catboost():
    f0 = _test_evaluate(model_cls=CatBoost, fold=0)
    f1 = _test_evaluate(model_cls=CatBoost, fold=0)
    expected_score = 0.6595
    assert f0 == f1 == pytest.approx(expected_score, abs=1e-4)
    f0 = _test_evaluate(model_cls=CatBoost, fold=1)
    f1 = _test_evaluate(model_cls=CatBoost, fold=1)
    expected_score = 0.6852
    assert f0 == f1 == pytest.approx(expected_score, abs=1e-4)
