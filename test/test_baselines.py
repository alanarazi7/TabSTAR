import json
import pytest
from collections import namedtuple

from do_baseline_refactored import run_baseline_benchmark
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.baselines.random_forest import RandomForest

BenchmarkParams = namedtuple("BenchmarkParams", ["model", "run_num", "expected_metric", "dataset_id", "model_cls"])

PARAMS = [
    BenchmarkParams(
        model="rf",
        run_num=0,
        expected_metric=0.56,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
        model_cls=RandomForest,
    ),
    BenchmarkParams(
        model="xgb",
        run_num=0,
        expected_metric=0.56,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
        model_cls=XGBoost,
    ),
    BenchmarkParams(
        model="xgb",
        run_num=0,
        expected_metric=0.56,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING,
        model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            expected_metric=0.56,
            dataset_id=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
            model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            expected_metric=0.56,
            dataset_id=OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT,
            model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            expected_metric=0.56,
            dataset_id=OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW,
            model_cls=XGBoost,
    )
    # Add more BenchmarkParams(...) as needed
]


def setup_benchmark_env(tmp_path, monkeypatch):
    results_dir = tmp_path / ".benchmark_results"
    results_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    return results_dir


@pytest.mark.parametrize("params", PARAMS)
def test_run_benchmarks_function(monkeypatch, tmp_path, params):
    """
    tests run_xgboost_benchmark with parameters from PARAMS. It sets up the environment,
    runs the benchmark, checks that it creates a valid result file in .benchmark_results and verifies the metric is consistent.
    """
    results_dir = setup_benchmark_env(tmp_path, monkeypatch)
    class Args:  # TODO I think this should be removed, will discuss in PR
        train_examples = 10000  # or any default you want

    run_baseline_benchmark(params.model_cls, params.dataset_id, params.run_num, Args)
    result_files = list(results_dir.glob("*.txt"))

    with open(result_files[0], "r") as f:
        data = json.load(f)
    print(f"Result data: {data}")
    assert round(data["metric"], 2) == params.expected_metric
    
