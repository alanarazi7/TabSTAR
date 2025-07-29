import math
from dataclasses import dataclass, asdict

import numpy as np
import wandb
from optuna import Trial
from xgboost import XGBRegressor, XGBClassifier

from tabstar_paper.benchmarks.constants import CPU_CORES
from tabular.constants import VERBOSE, OPTUNA_BUDGET
from tabular.evaluation.cross_validation import get_kfold_splitter, get_optuna_study, make_train_dev_splits
from tabular.evaluation.metrics import calculate_metric
from tabular.evaluation.sklearn_model import init_model
from tabular.models.abstract_sklearn import TabularSklearnModel
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask
from tabular.preprocessing.target import standardize_y_train_test, transform_target, fit_standard_scaler
from tabular.preprocessing.trees.categorical import fit_encode_categorical, transform_encoder_categorical
from tabular.preprocessing.trees.numerical import fill_median
from tabular.utils.utils import verbose_print



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


class XGBoostOptuna(TabularSklearnModel):
    MODEL_NAME = f"XGBoost-Opt{OPTUNA_BUDGET} 🍃"
    SHORT_NAME = "xgbopt"
    PROCESSING = PreprocessingMethod.TREES_OPT

    def initialize_model(self):
        self.model = init_model(config=self.config, is_reg=self.dataset.is_regression,
                                classifier_cls=XGBClassifier, regressor_cls=XGBRegressor)


    def train(self):
        self.model = None
        self.config = None
        assert all(v is None for v in [self.config, self.model, self.y_scaler, self.x_median])
        print(f"Starting Optuna study for {self.dataset.sid}")
        study = get_optuna_study()
        study.optimize(self.objective, n_jobs=CPU_CORES, timeout=OPTUNA_BUDGET)
        print(f"Done studying, did {len(study.trials)} runs 🤓")
        self.config = XGBoostTunedHyperparams(**study.best_params)
        wandb.log({"optuna_best_params": asdict(self.config), "optuna_n_trials": len(study.trials)})
        print(f"✅ Best params: {self.config}")
        self.initialize_model()
        assert self.model is not None
        x_train, y_train = self.load_train()
        if self.task_type == SupervisedTask.REGRESSION:
            self.y_scaler = fit_standard_scaler(y_train)
            y_train = transform_target(y_train, transformer=self.y_scaler)
        for col in self.dataset.numerical_col_names:
            median_filler = fill_median(x_train=x_train[col])
            x_train[col] = median_filler.src
            if self.x_median is None:
                self.x_median = {}
            self.x_median[col] = median_filler.median
        for col in self.dataset.cat_bool_col_names:
            cat_encoder = fit_encode_categorical(s=x_train[col])
            x_train[col] = transform_encoder_categorical(s=x_train[col], encoder=cat_encoder)
            if self.x_encoder is None:
                self.x_encoder = {}
            self.x_encoder[col] = cat_encoder
        self.model.fit(x_train, y_train, verbose=VERBOSE)

    def objective(self, trial: Trial) -> float:
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
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample,
            colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, min_child_weight=min_child_weight,
            alpha=alpha, reg_lambda=reg_lambda, gamma=gamma)

        x, y = self.load_train()
        splitter = get_kfold_splitter(is_regression=self.dataset.is_regression)
        fold_scores = []
        for f, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            verbose_print(f"Fold {f}")
            x_train, y_train, x_dev, y_dev = make_train_dev_splits(x=x, y=y, train_idx=train_idx, val_idx=val_idx)
            for col in self.dataset.numerical_col_names:
                verbose_print(f"Numerical column: {col}")
                median_filler = fill_median(x_train=x_train[col], x_test=x_dev[col])
                x_train[col] = median_filler.src
                x_dev[col] = median_filler.target
            for col in self.dataset.cat_bool_col_names:
                verbose_print(f"🐈 Processing Categorical column: {col}")
                cat_encoder = fit_encode_categorical(s=x_train[col])
                x_train[col] = transform_encoder_categorical(s=x_train[col], encoder=cat_encoder)
                x_dev[col] = transform_encoder_categorical(s=x_dev[col], encoder=cat_encoder)
            if self.task_type == SupervisedTask.REGRESSION:
                y_train, y_dev = standardize_y_train_test(y_train, y_dev)

            model = init_model(config=trial_config, is_reg=self.dataset.is_regression,
                               classifier_cls=XGBClassifier, regressor_cls=XGBRegressor)
            model.fit(x_train, y_train, verbose=VERBOSE)
            preds = self.predict_from_model(x_dev, model=model)
            score = calculate_metric(self.task_type, y_dev, preds)
            fold_scores.append(score)

        return float(np.mean(fold_scores))