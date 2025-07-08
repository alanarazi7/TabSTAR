import json
import pytest
from collections import namedtuple

from do_baseline_refactored import run_baseline_benchmark
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.baselines.xgboost import XGBoost
from tabstar_paper.baselines.random_forest import RandomForest

SCORE_TOLERANCE = 0.15
MIN_FACTOR = 1 - SCORE_TOLERANCE
MAX_FACTOR = 1 + SCORE_TOLERANCE

BenchmarkParams = namedtuple("BenchmarkParams", ["model", "run_num", "min_score", "max_score", "dataset_id", "model_cls"])

PARAMS = [ # the min and max scores are based on the results of the paper
    BenchmarkParams(
        model="rf",
        run_num=0,
        min_score=0.8738,
        max_score=0.9441,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
        model_cls=RandomForest,
    ),
    BenchmarkParams(
        model="xgb",
        run_num=0,
        min_score=0.8902,
        max_score=0.9447,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
        model_cls=XGBoost,
    ),
    BenchmarkParams(
        model="xgb",
        run_num=0,
        min_score=0.6766,
        max_score=0.7207,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_KICKSTARTER_FUNDING,
        model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            min_score=0.7330,
            max_score=0.8730,
            dataset_id=OpenMLDatasetID.BIN_SOCIAL_IMDB_GENRE_PREDICTION,
            model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            min_score=0.8543,
            max_score=0.9153,
            dataset_id=OpenMLDatasetID.MUL_CONSUMER_PRODUCT_SENTIMENT,
            model_cls=XGBoost,
    ),
    BenchmarkParams(
            model="xgb",
            run_num=0,
            min_score=0.8771,
            max_score=0.9013,
            dataset_id=OpenMLDatasetID.MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW,
            model_cls=XGBoost,
    )
]


def setup_benchmark_env(tmp_path, monkeypatch):
    results_dir = tmp_path / ".benchmark_results"
    results_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    return results_dir

def get_score_thresholds(params, calculated_score):
    """
    Returns the minimum and maximum score thresholds based on the provided parameters.
    """
    min_threshold = params.min_score * MIN_FACTOR
    max_threshold = min(params.max_score * MAX_FACTOR, 1.0)
    return min_threshold, max_threshold

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
    calculated_score = data["metric"]
    min_threshold, max_threshold = get_score_thresholds(params, calculated_score)

    assert min_threshold < calculated_score < max_threshold, f"Score {calculated_score} not within range [{min_threshold}, {max_threshold}]"
