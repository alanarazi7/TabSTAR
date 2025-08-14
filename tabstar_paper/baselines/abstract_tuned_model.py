from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Dict, Type, Any

import numpy as np
import torch
from optuna import create_study, Trial
from optuna.samplers import RandomSampler
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold, KFold

from tabstar.constants import SEED
from tabstar.training.devices import CPU_CORES
from tabstar_paper.baselines.abstract_model import TabularModel
from tabstar_paper.constants import TIME_BUDGET
from tabstar_paper.datasets.objects import SupervisedTask

CV_INNER_FOLDS = 5

class TunedTabularModel(TabularModel):

    BASE_CLS: Type[TabularModel] = None
    HYPERPARAM_CLS: dataclass = None

    def __init__(self, problem_type: SupervisedTask, device: torch.device, verbose: bool = False, **kwargs):
        super().__init__(problem_type=problem_type, device=device, verbose=verbose, **kwargs)
        self.optuna_dict: Dict[str, float] = {"time_budget": TIME_BUDGET}
        self.USE_VAL_SPLIT = False
        self.USE_MEDIAN_FILLING = self.BASE_CLS.USE_MEDIAN_FILLING
        self.USE_CATEGORICAL_ENCODING = self.BASE_CLS.USE_CATEGORICAL_ENCODING
        self.USE_TEXT_EMBEDDINGS = self.BASE_CLS.USE_TEXT_EMBEDDINGS

    def initialize_model(self):
        # As opposed to the base class, we will initialize the model after the hyperparameters are tuned.
        return None

    def initialize_tuned_model(self, params: Dict[str, Any]):
        raise NotImplementedError("This method should be implemented in the subclass to initialize the tuned model.")

    def fit_tuned_model(self, model: Any, x_train: DataFrame, y_train: Series, ):
        raise NotImplementedError("This method should be implemented in the subclass to fit the tuned model.")

    def fit_fold_model(self, model: Any, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series) -> float:
        raise NotImplementedError("This method should be implemented in the subclass to fit the model on a fold.")

    def fit(self, x: DataFrame, y: Series):
        print(f"Starting Optuna study for model {self.MODEL_NAME}")
        # We always maximize even for regression, as we use R^2
        assert self.model_ is None
        study = create_study(direction="maximize", sampler=RandomSampler(seed=SEED))
        objective_with_data = partial(self.objective, x=x, y=y)
        study.optimize(objective_with_data, n_jobs=CPU_CORES, timeout=TIME_BUDGET)
        best_params = dict(study.best_params)
        print(f"Done studying, did {len(study.trials)} runs 🤓\n Best params: {best_params}")
        self.optuna_dict.update({"optuna_best_params": best_params, "optuna_n_trials": len(study.trials)})
        best_params = self.HYPERPARAM_CLS()
        assert self.model_ is None
        self.model_ = self.initialize_tuned_model(params=best_params)
        # TODO: We are refitting the model, but we could do bagging and predict over the folds like in TabArena
        x_train, y_train = x.copy(), y.copy()
        self.fit_preprocessor(x_train=x_train, y_train=y_train)
        x_train, y_train = self.transform_preprocessor(x=x_train, y=y_train)
        self.fit_tuned_model(model=self.model_, x_train=x_train, y_train=y_train)

    def get_trial_config(self, trial: Trial) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented in the subclass to return trial configuration.")

    def objective(self, trial: Trial, x: DataFrame, y: Series) -> float:
        trial_config = self.get_trial_config(trial=trial)
        fold_scores = []
        if self.is_cls:
            splitter = StratifiedKFold(n_splits=CV_INNER_FOLDS, shuffle=True, random_state=SEED)
        else:
            splitter = KFold(n_splits=CV_INNER_FOLDS, shuffle=True, random_state=SEED)
        for f, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            x_train = x.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            x_val = x.iloc[val_idx].copy()
            y_val = y.iloc[val_idx].copy()
            model_obj = deepcopy(self)
            fold_model = model_obj.initialize_tuned_model(params=trial_config)
            model_obj.fit_preprocessor(x_train=x_train, y_train=y_train)
            x_train, y_train = model_obj.transform_preprocessor(x=x_train, y=y_train)
            x_val, y_val = model_obj.transform_preprocessor(x=x_val, y=y_val)
            fold_score = model_obj.fit_fold_model(model=fold_model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
            print(f"Trial num {trial.number}, Fold {f}, score: {fold_score}")
            fold_scores.append(fold_score)
        avg_loss = float(np.mean(fold_scores))
        return avg_loss
