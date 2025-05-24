from dataclasses import dataclass, asdict

import numpy as np
import wandb
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoostError
from optuna import Trial

from tabular.constants import VERBOSE, OPTUNA_CPU, OPTUNA_BUDGET
from tabular.evaluation.cross_validation import get_kfold_splitter, get_optuna_study, make_train_dev_splits
from tabular.evaluation.metrics import calculate_metric
from tabular.evaluation.sklearn_model import init_model
from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask
from tabular.preprocessing.target import standardize_y_train_test, fit_standard_scaler, transform_target
from tabular.utils.utils import verbose_print, cprint

LOG_LEVEL = "Verbose" if VERBOSE else "Silent"

@dataclass
class CatBoostDefaultHyperparams:
    early_stopping_rounds: int = 50
    iterations: int = 2000
    od_pval: float = 0.001
    thread_count = 1

@dataclass
class CatBoostTunedHyperparams:
    iterations: int
    learning_rate: float
    random_strength: int
    l2_leaf_reg: float
    bagging_temperature: float
    leaf_estimation_iterations: int
    # Thread count is set to one since it seems is better to parallelize the trials, not the internal runs
    thread_count = 1


class CatBoost(TabularSklearnModel):

    MODEL_NAME = "CatBoost 😸"
    SHORT_NAME = "cat"
    PROCESSING = PreprocessingMethod.CATBOOST

    def initialize_model(self):
        self.model = init_model(config=self.config, is_reg=self.dataset.is_regression,
                                classifier_cls=CatBoostClassifier, regressor_cls=CatBoostRegressor)

    def train(self):
        verbose_print(f"Training {self.MODEL_NAME} model for dataset {self.dataset.sid}")
        x_train, y_train, x_dev, y_dev = self.load_all()
        cprint(f"Training {self.MODEL_NAME} over {len(x_train)} examples. Dev set has {len(x_dev)} examples")
        self.model.fit(x_train, y_train, eval_set=(x_dev, y_dev), logging_level=LOG_LEVEL, use_best_model=True,
                       cat_features=self.dataset.cat_col_indices)

    def set_config(self) -> CatBoostDefaultHyperparams:
        # Using the "default" hyperparameters of the FT-Transformer paper: https://arxiv.org/pdf/2106.11959
        return CatBoostDefaultHyperparams()

# TODO: could be generalized to a 'TabularOptunaModel' wrapper
class CatBoostOptuna(CatBoost):
    MODEL_NAME = f"CatBoost-Opt{OPTUNA_BUDGET} 😼"
    SHORT_NAME = "catopt"
    PROCESSING = PreprocessingMethod.CATBOOST_OPT

    def train(self):
        self.config = None
        self.model = None
        cprint(f"Starting Optuna study for {self.dataset.sid}")
        study = get_optuna_study()
        study.optimize(self.objective, n_jobs=OPTUNA_CPU, timeout=OPTUNA_BUDGET,
                       catch=(CatBoostError,))
        cprint(f"Done studying, did {len(study.trials)} runs 🤓")
        self.config = CatBoostTunedHyperparams(**study.best_params)
        wandb.log({"optuna_best_params": asdict(self.config), "optuna_n_trials": len(study.trials)})
        cprint(f"✅ Best params: {self.config}")
        self.initialize_model()
        assert self.model is not None
        verbose_print(f"Training {self.MODEL_NAME} FULL model for dataset {self.dataset.sid}")
        x_train, y_train = self.load_train()
        if self.task_type == SupervisedTask.REGRESSION:
            self.y_scaler = fit_standard_scaler(y_train)
            y_train = transform_target(y_train, transformer=self.y_scaler)
        self.model.fit(x_train, y_train, logging_level=LOG_LEVEL, use_best_model=True,
                       cat_features=self.dataset.cat_col_indices)

    def objective(self, trial: Trial) -> float:
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

        x, y = self.load_train()
        splitter = get_kfold_splitter(is_regression=self.dataset.is_regression)
        fold_scores = []
        for f, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            verbose_print(f"Training fold {f}")
            x_train, y_train, x_dev, y_dev = make_train_dev_splits(x=x, y=y, train_idx=train_idx, val_idx=val_idx)
            if self.task_type == SupervisedTask.REGRESSION:
                y_train, y_dev = standardize_y_train_test(y_train, y_dev)
            fold_model = init_model(config=trial_config, is_reg=self.dataset.is_regression,
                                    classifier_cls=CatBoostClassifier, regressor_cls=CatBoostRegressor)
            fold_model.fit(x_train, y_train, eval_set=(x_dev, y_dev), logging_level=LOG_LEVEL, use_best_model=True,
                           cat_features=self.dataset.cat_col_indices)
            predictions = self.predict_from_model(x_dev, model=fold_model)
            metric = calculate_metric(task_type=self.task_type, y_true=y_dev, y_pred=predictions)
            fold_scores.append(metric)
        avg_score = float(np.mean(fold_scores))
        return avg_score
