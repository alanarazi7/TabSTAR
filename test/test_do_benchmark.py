import json
import pytest
from collections import namedtuple

from do_benchmark import run_benchmark
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar_paper.baselines.xgboost import XGBoost

BenchmarkParams = namedtuple("BenchmarkParams", ["model", "run_num", "expected_metric", "dataset_id", "model_cls"])

PARAMS = [
    BenchmarkParams(
        model="xgb",
        run_num=0,
        expected_metric=0.56,
        dataset_id=OpenMLDatasetID.BIN_PROFESSIONAL_FAKE_JOB_POSTING,
        model_cls=XGBoost,
    ),
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
    tests run_benchmark with parameters from PARAMS. It sets up the environment,
    runs the benchmark, checks that it creates a valid result file in .benchmark_results and verifies the metric is consistent.
    """
    results_dir = setup_benchmark_env(tmp_path, monkeypatch)

    class Args:  # TODO I think this should be removed, will discuss in PR
        train_examples = 10000  # or any default you want

    run_benchmark(params.model_cls, params.dataset_id, params.run_num, Args)

    result_files = list(results_dir.glob("*.txt"))
    assert result_files, f"No benchmark result file was created for run_num={params.run_num}"

    with open(result_files[0], "r") as f:
        data = json.load(f)
    assert round(data["metric"], 2) == params.expected_metric
