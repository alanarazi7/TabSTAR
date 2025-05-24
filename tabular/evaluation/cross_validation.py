from typing import List

from optuna import create_study, Study
from optuna.samplers import RandomSampler
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold

from tabular.utils.utils import SEED

N_SPLITS = 5


def get_kfold_splitter(is_regression: bool) -> KFold | StratifiedKFold:
    if is_regression:
        return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    else:
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def get_optuna_study() -> Study:
    # We always maximize even for regression, as we use R^2
    study = create_study(direction="maximize", sampler=RandomSampler(seed=SEED))
    return study


def make_train_dev_splits(x: DataFrame, y: Series, train_idx: List[int], val_idx: List[int]):
    x_train = x.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    x_dev   = x.iloc[val_idx].copy()
    y_dev   = y.iloc[val_idx].copy()
    return x_train, y_train, x_dev, y_dev