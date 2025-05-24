from typing import List

import openml


def get_suite_datasets(sid: int, name: str = "", n_datasets: int = 0) -> List[int]:
    benchmark_suite = openml.study.get_suite(sid)
    if name and (benchmark_suite.name != name):
        raise ValueError(f"Expected {name=}, but got {benchmark_suite.name=}")
    dataset_ids = benchmark_suite.data
    if n_datasets and (len(dataset_ids) != n_datasets):
        raise ValueError(f"Expected {n_datasets=}, but got {len(dataset_ids)=}")
    return list(dataset_ids)