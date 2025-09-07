import os
from typing import Type

import pytest
import torch

# from tabstar.constants import SEED
from tabstar.datasets.all_datasets import OpenMLDatasetID
# from tabstar.training.utils import fix_seed
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.catboost import CatBoost
from tabstar_paper.benchmarks.evaluate import evaluate_on_dataset

# # Ensure deterministic behavior across environments
# os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# # Additional environment variables for cross-platform determinism
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _test_evaluate(model_cls: Type[TabularModel], fold: int = 0) -> float:
    # # Fix all random seeds before each evaluation for complete determinism
    # fix_seed(SEED)
    d_summary = evaluate_on_dataset(model_cls=model_cls, dataset_id=OpenMLDatasetID.BIN_FINANCIAL_CREDIT_GERMAN,
                                    fold=fold, device=torch.device("cpu"), train_examples=100, verbose=False)
    return d_summary["test_score"]




def test_catboost():
    f0 = _test_evaluate(model_cls=CatBoost, fold=0)
    f1 = _test_evaluate(model_cls=CatBoost, fold=0)
    expected_score = 0.7314
    assert f0 == f1 == pytest.approx(expected_score, abs=1e-4)
    f0 = _test_evaluate(model_cls=CatBoost, fold=1)
    f1 = _test_evaluate(model_cls=CatBoost, fold=1)
    expected_score = 0.6062
    assert f0 == f1 == pytest.approx(expected_score, abs=1e-4)
