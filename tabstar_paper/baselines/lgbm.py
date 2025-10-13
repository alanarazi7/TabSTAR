from dataclasses import dataclass
from typing import Dict, Any

from lightgbm import LGBMClassifier, LGBMRegressor
from optuna import Trial
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.abstract_tuned_model import TunedTabularModel


def init_lgbm(is_cls: bool, params: dict) -> LGBMRegressor | LGBMClassifier:
    model_cls = LGBMClassifier if is_cls else LGBMRegressor
    return model_cls(**params)



class LightGBM(TabularModel):

    MODEL_NAME = "LightGBM 💡"
    SHORT_NAME = "light"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> LGBMRegressor | LGBMClassifier:
        return init_lgbm(is_cls=self.is_cls, params={"verbose": -1, "random_state": SEED})

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=self.categorical_indices)


@dataclass
class LightGBMTunedHyperparams:
    num_leaves: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    random_state: int = SEED
    n_jobs: int = 1
    nthread: int = 1
    verbose = -1


class LightGBMOpt(TunedTabularModel):

    MODEL_NAME = "LightGBM-Opt ⚡"
    SHORT_NAME = "lightopt"
    REFIT_REQUIRES_VAL = False
    BASE_CLS = LightGBM

    def initialize_tuned_model(self, params: Dict[str, Any], is_last_model: bool = False):
        assert self.model_ is None, "Model should be None before initializing tuned model."
        if is_last_model:
            params["nthread"] = CPU_CORES
        self.model_ = init_lgbm(is_cls=self.is_cls, params=params)

    def fit_tuned_model(self, x_train: DataFrame, y_train: Series):
        self.model_.fit(x_train, y_train, categorical_feature=self.categorical_indices)

    def fit_fold_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=self.categorical_indices)

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        num_leaves = trial.suggest_int("num_leaves", 5, 50)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 2000)
        min_child_weights = [1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
        min_child_weight = trial.suggest_categorical("min_child_weight", min_child_weights)
        subsample = trial.suggest_float("subsample", 0.2, 0.8)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 0.8)
        reg_alpha = trial.suggest_categorical("reg_alpha", [0, 0.1, 1, 2, 5, 7, 10, 50, 100])
        reg_lambda = trial.suggest_categorical("reg_lambda", [0, 0.1, 1, 5, 10, 20, 50, 100])
        params = LightGBMTunedHyperparams(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        return vars(params)