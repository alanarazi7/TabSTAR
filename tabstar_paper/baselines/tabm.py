from dataclasses import dataclass
from typing import Dict, Any

from optuna import Trial
from pandas import DataFrame, Series
from pytabkit import TabM_D_Classifier, TabM_D_Regressor

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.abstract_tuned_model import TunedTabularModel


def init_tabm(is_cls: bool, params: dict) -> TabM_D_Regressor | TabM_D_Classifier:
    for k in ['use_dropout', 'use_weight_decay']:
        if k in params:
            params.pop(k)
    model_cls = TabM_D_Classifier if is_cls else TabM_D_Regressor
    return model_cls(**params)


@dataclass
class TabMDefaultHyperparams:
    device: str
    random_state: int = SEED
    n_threads: int = CPU_CORES


class TabM(TabularModel):
    MODEL_NAME = "TabM Ⓜ️"
    SHORT_NAME = "tabm"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> TabM_D_Classifier | TabM_D_Regressor:
        params = TabMDefaultHyperparams(device=str(self.device))
        model = init_tabm(is_cls=self.is_cls, params=vars(params))
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)


@dataclass
class TabMTunedHyperparams:
    device: str
    n_blocks: int
    d_block: int
    dropout: float
    lr: float
    weight_decay: float
    n_threads: int
    random_state: int = SEED


class TabMOpt(TunedTabularModel):

    MODEL_NAME = "TabM-Opt 〽️"
    SHORT_NAME = "tabmopt"
    REFIT_REQUIRES_VAL = True
    BASE_CLS = TabM

    def initialize_tuned_model(self, params: Dict[str, Any], is_last_model: bool = False):
        assert self.model_ is None, "Model should be None before initializing tuned model."
        if is_last_model:
            params["n_threads"] = CPU_CORES
        self.model_ = init_tabm(is_cls=self.is_cls, params=params)

    def fit_tuned_model(self, x_train: DataFrame, y_train: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, cat_col_names=cat_col_names)

    def fit_fold_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)

    def add_missing_params(self, study: Any, best_params: Dict[str, Any]):
        for k in {'dropout', 'weight_decay'}:
            if k not in best_params:
                best_params[k] = study.best_trial.user_attrs[k]

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        d_block = trial.suggest_int("d_block", 64, 1024)

        # dropout: toggle + value (0.0 allowed, but only used if toggled on)
        use_dropout = trial.suggest_categorical("use_dropout", [False, True])
        dropout = 0.0 if not use_dropout else trial.suggest_float("dropout", 0.0, 0.5)
        trial.set_user_attr('dropout', dropout)

        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        # weight decay: toggle + positive value (log scale can’t include 0)
        use_weight_decay = trial.suggest_categorical("use_weight_decay", [False, True])
        weight_decay = 0.0 if not use_weight_decay else trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

        trial.set_user_attr('weight_decay', weight_decay)
        trial.set_user_attr("device", str(self.device))
        trial_config = TabMTunedHyperparams(n_blocks=n_blocks,
                                            d_block=d_block,
                                            dropout=dropout,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            device=str(self.device),
                                            n_threads=1)
        return vars(trial_config)
