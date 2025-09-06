from dataclasses import dataclass
from typing import Dict, Any, Set, List

from catboost import CatBoostRegressor, CatBoostClassifier
from optuna import Trial
from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.baselines.abstract_tuned_model import TunedTabularModel


def init_catboost(is_cls: bool, params: dict) -> CatBoostRegressor | CatBoostClassifier:
    model_cls = CatBoostClassifier if is_cls else CatBoostRegressor
    return model_cls(**params)

def get_cat_features(x_train: DataFrame, categorical_features: Set) -> List[int]:
    return [i for i, c in enumerate(x_train.columns) if c in categorical_features]


@dataclass
class CatBoostDefaultHyperparams:
    # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
    early_stopping_rounds: int = 50
    iterations: int = 2000
    od_pval: float = 0.001
    random_state: int = SEED
    thread_count = 1  # Force single thread for deterministic results across environments

class CatBoost(TabularModel):

    MODEL_NAME = "CatBoost 😸"
    SHORT_NAME = "cat"
    USE_VAL_SPLIT = True
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = True

    def initialize_model(self) -> CatBoostRegressor | CatBoostClassifier:
        return init_catboost(is_cls=self.is_cls, params=vars(CatBoostDefaultHyperparams()))

    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_features = get_cat_features(x_train=x_train, categorical_features=self.categorical_features)
        logging_level = "Verbose" if self.verbose else "Silent"
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features,
                        logging_level=logging_level)


@dataclass
class CatBoostTunedHyperparams:
    iterations: int
    learning_rate: float
    random_strength: int
    l2_leaf_reg: float
    bagging_temperature: float
    leaf_estimation_iterations: int
    # Thread count is set to one since it seems is better to parallelize the trials, not the internal runs
    thread_count: int = 1


class CatBoostOpt(TunedTabularModel):

    MODEL_NAME = f"CatBoost-Opt 😼"
    SHORT_NAME = "catopt"
    BASE_CLS = CatBoost

    def initialize_tuned_model(self, params: Dict[str, Any], is_last_model: bool = False):
        assert self.model_ is None, "Model should be None before initializing tuned model."
        if is_last_model:
            params["thread_count"] = CPU_CORES
        self.model_ = init_catboost(is_cls=self.is_cls, params=params)

    def fit_tuned_model(self, x_train: DataFrame, y_train: Series):
        cat_features = get_cat_features(x_train=x_train, categorical_features=self.categorical_features)
        self.model_.fit(x_train, y_train, cat_features=cat_features, logging_level="Silent")

    def fit_fold_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        cat_features = get_cat_features(x_train=x_train, categorical_features=self.categorical_features)
        self.model_.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True, cat_features=cat_features,
                        logging_level="Silent")

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        # Hyperparam search as suggested by TabPFN-v2 paper: https://www.nature.com/articles/s41586-024-08328-6.pdf
        lr = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
        random_strength = trial.suggest_int("random_strength", 1, 20)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 20)
        iterations = trial.suggest_int("iterations", 100, 4000)
        trial_config = CatBoostTunedHyperparams(iterations=iterations,
                                                learning_rate=lr,
                                                random_strength=random_strength,
                                                l2_leaf_reg=l2_leaf_reg,
                                                bagging_temperature=bagging_temperature,
                                                leaf_estimation_iterations=leaf_estimation_iterations)
        return vars(trial_config)