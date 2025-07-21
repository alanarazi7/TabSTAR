import os
from os.path import join

from tabstar_paper.pretraining.paths import pretrain_exp_dir

CACHE_DIR = ".tabular_cache"

_BASELINES_DIR = join(CACHE_DIR, "baselines")
_DATASET_DIR = join(CACHE_DIR, "datasets")


def get_model_path(run_name: str) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, "best_checkpoint")

def get_checkpoint(run_name: str, epoch: int) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, f"checkpoint_{epoch}")


def train_results_path(run_name: str) -> str:
    return join(_BASELINES_DIR, run_name, "results.json")


def dataset_run_properties_dir(run_num: int, train_examples: int) -> str:
    return join(_DATASET_DIR, f"run{run_num}_n{train_examples}")

def properties_path(data_dir: str) -> str:
    return join(data_dir, "properties.json")

def create_dir(path: str, is_file: bool = False):
    if is_file:
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)
