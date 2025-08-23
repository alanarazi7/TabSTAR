from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from optuna import Trial
from pandas import DataFrame, Series
from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.abstract_tuned_model import TunedTabularModel
from tabstar_paper.datasets.objects import SupervisedTask


def init_realmlp(is_cls: bool, params: dict) -> RealMLP_TD_Regressor | RealMLP_TD_Classifier:
    model_cls = RealMLP_TD_Classifier if is_cls else RealMLP_TD_Regressor
    return model_cls(**params)

def get_val_metric_name(problem_type: SupervisedTask) -> str:
    task2metric = {SupervisedTask.BINARY: 'cross_entropy',
                   SupervisedTask.MULTICLASS: '1-auc_ovr',
                   SupervisedTask.REGRESSION: 'rmse'}
    return task2metric[problem_type]


@dataclass
class RealMlpDefaultHyperparams:
    device: str
    val_metric_name: str
    random_state: int = SEED
    n_threads: int = CPU_CORES
    use_ls: bool = False

class RealMLP(TabularModel):
    MODEL_NAME = "RealMLP 🕸"
    SHORT_NAME = "real"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = True
    USE_CATEGORICAL_ENCODING = True
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> RealMLP_TD_Classifier | RealMLP_TD_Regressor:
        val_metric = get_val_metric_name(self.problem_type)
        params = RealMlpDefaultHyperparams(device=str(self.device), val_metric_name=val_metric)
        model = init_realmlp(is_cls=self.is_cls, params=vars(params))
        return model

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)


@dataclass
class RealMLPTunedHyperparams:
    device: str
    val_metric_name: str
    num_emb_type: str
    add_front_scale: bool
    lr: float
    p_drop: float
    act: str
    hidden_sizes: List[int]
    wd: float
    plr_sigma: float
    use_ls: bool = False
    n_threads: int = 1
    random_state: int = SEED


class RealMlpOpt(TunedTabularModel):

    MODEL_NAME = "RealMLP-Opt 🌐"
    SHORT_NAME = "realopt"
    BASE_CLS = RealMLP

    def initialize_tuned_model(self, params: Dict[str, Any], is_last_model: bool = False):
        assert self.model_ is None, "Model should be None before initializing tuned model."
        if is_last_model:
            params["n_threads"] = CPU_CORES
        self.model_ = init_realmlp(is_cls=self.is_cls, params=params)

    def fit_tuned_model(self, x_train: DataFrame, y_train: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, cat_col_names=cat_col_names)

    def fit_fold_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_col_names = list(self.categorical_features)
        self.model_.fit(x_train, y_train, x_val, y_val, cat_col_names=cat_col_names)

    def add_missing_params(self, study: Any, best_params: Dict[str, Any]):
        for k in {'val_metric_name', 'add_front_scale', 'p_drop', 'hidden_sizes'}:
            best_params[k] = study.best_trial.user_attrs[k]

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        val_metric_name = get_val_metric_name(self.problem_type)
        trial.set_user_attr('val_metric_name', val_metric_name)
        num_emb_type = trial.suggest_categorical("num_emb_type", ["none", "pl", "plr", "pbld"])
        lr = trial.suggest_float("lr", 2e-2, 3e-1, log=True)

        dropout_value_probability = np.random.uniform(0, 1)
        if dropout_value_probability < 0.3:
            p_drop = 0.0
        elif dropout_value_probability < 0.8:
            p_drop = 0.15
        else:
            p_drop = 0.3
        trial.set_user_attr("p_drop", p_drop)

        act = trial.suggest_categorical("act", ["relu", "selu", "mish"])
        wd = trial.suggest_categorical("wd", [0.0, 2e-2])

        hidden_sizes_probability = np.random.uniform(0, 1)
        if hidden_sizes_probability < 0.6:
            hidden_sizes = [256, 256, 256]
        elif hidden_sizes_probability < 0.8:
            hidden_sizes = [64, 64, 64, 64, 64]
        else:
            hidden_sizes = [512]
        trial.set_user_attr('hidden_sizes', hidden_sizes)

        scaling_probability = np.random.uniform(0, 1)
        if scaling_probability < 0.6:
            add_front_scale = True
        else:
            add_front_scale = False
        trial.set_user_attr('add_front_scale', add_front_scale)

        plr_sigma = trial.suggest_float("plr_sigma", 0.05, 0.5, log=True)

        trial_config = RealMLPTunedHyperparams(device=str(self.device),
                                               val_metric_name=val_metric_name,
                                               num_emb_type=num_emb_type,
                                               add_front_scale=add_front_scale,
                                               lr=lr,
                                               p_drop=p_drop,
                                               act=act,
                                               hidden_sizes=hidden_sizes,
                                               wd=wd,
                                               plr_sigma=plr_sigma)

        return vars(trial_config)
