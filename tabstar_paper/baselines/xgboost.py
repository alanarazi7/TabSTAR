import math
from dataclasses import dataclass
from typing import Dict, Any

from optuna import Trial
from xgboost import XGBRegressor, XGBClassifier
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.abstract_tuned_model import TunedTabularModel


def init_xgboost(is_cls: bool, params: Dict[str, Any]) -> XGBRegressor | XGBClassifier:
    model_cls = XGBClassifier if is_cls else XGBRegressor
    return model_cls(**params)


@dataclass
class XGBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    booster: str = "gbtree"
    random_state: int = SEED


class XGBoost(TabularModel):

    MODEL_NAME = "XGBoost 🌲"
    SHORT_NAME = "xgb"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> XGBRegressor | XGBClassifier:
        params = XGBoostDefaultHyperparams()
        return init_xgboost(self.is_cls, params=vars(params))

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        self.model_.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)


@dataclass
class XGBoostTunedHyperparams:
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    min_child_weight: float
    alpha: float
    reg_lambda: float
    gamma: float
    n_jobs: int = 1
    nthread: int = 1


class XGBoostOpt(TunedTabularModel):

    MODEL_NAME = f"XGBoost-Opt 🍃"
    SHORT_NAME = "xgbopt"
    BASE_CLS = XGBoost

    def initialize_tuned_model(self, params: Dict[str, Any]):
        return init_xgboost(is_cls=self.is_cls, params=params)

    def fit_tuned_model(self, model: Any, x_train: DataFrame, y_train: Series):
        model.fit(x_train, y_train, verbose=self.verbose)

    def fit_fold_model(self, model: Any, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series) -> float:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=self.verbose)
        score = model.score(x_val, y_val)
        return score

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        # Hyperparam search as suggested by TabPFN-v2 paper: https://www.nature.com/articles/s41586-024-08328-6.pdf
        n_estimators = trial.suggest_int("n_estimators", 100, 4000)
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        subsample = trial.suggest_float("subsample", 0.2, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.2, 1.0)
        min_child_weight = trial.suggest_float("min_child_weight", math.exp(-16), math.exp(5), log=True)
        alpha = trial.suggest_float("alpha", math.exp(-16), math.exp(2), log=True)
        reg_lambda = trial.suggest_float("reg_lambda", math.exp(-16), math.exp(2), log=True)
        gamma = trial.suggest_float("gamma", math.exp(-16), math.exp(2), log=True)
        trial_config = XGBoostTunedHyperparams(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            min_child_weight=min_child_weight,
            alpha=alpha,
            reg_lambda=reg_lambda,
            gamma=gamma)
        return vars(trial_config)