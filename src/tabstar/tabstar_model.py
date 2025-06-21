from typing import Optional, Tuple

import joblib
import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import autocast

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.datasets.benchmark_folds import get_tabstar_version
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.metrics import apply_loss_fn, calculate_metric
from tabstar.training.trainer import TabStarTrainer
from tabstar.training.utils import concat_predictions


class BaseTabSTAR:
    def __init__(self, verbose: bool = False,
                 device: Optional[str] = None,
                 pretrain_dataset: Optional[TabularDatasetID] = None):
        self.verbose = verbose
        self.preprocessor_: Optional[TabSTARVerbalizer] = None
        self.model_: Optional[PeftModel] = None
        self.device = get_device(device=device)
        print(f"🖥️ Using device: {self.device}")
        self.model_version = get_tabstar_version(pretrain_dataset=pretrain_dataset)

    def fit(self, X, y):
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        train_data, val_data = self._prepare_for_train(X, y)
        self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
        trainer = TabStarTrainer(device=self.device, model_version=self.model_version)
        trainer.train(train_data, val_data)
        trainer.load_model()
        self.model_ = trainer.model

    def predict(self, X):
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def is_cls(self) -> bool:
        raise NotImplementedError("Must be implemented in subclass")

    def save(self, path: str):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> 'BaseTabSTAR':
        return joblib.load(path)

    def _prepare_for_train(self, X, y) -> Tuple[TabSTARData, TabSTARData]:
        if not isinstance(X, DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(y, Series):
            raise ValueError("y must be a pandas Series.")
        raise_if_null_target(y)
        self.vprint(f"Preparing data for training. X shape: {X.shape}, y shape: {y.shape}")
        x_train, x_val, y_train, y_val = split_to_val(x=X, y=y, is_cls=self.is_cls)
        self.vprint(f"Split to validation set. Train has {len(x_train)} samples, validation has {len(x_val)} samples.")
        if self.preprocessor_ is None:
            self.preprocessor_ = TabSTARVerbalizer(is_cls=self.is_cls, verbose=self.verbose)
            self.preprocessor_.fit(x_train, y_train)
        train_data = self.preprocessor_.transform(x_train, y_train)
        self.vprint(f"Transformed training data: {train_data.x_txt.shape=}, x_num shape: {train_data.x_num.shape=}")
        val_data = self.preprocessor_.transform(x_val, y_val)
        return train_data, val_data

    def _infer(self, X) -> np.ndarray:
        data = self.preprocessor_.transform(X, y=None)
        dataloader = get_dataloader(data, is_train=False, batch_size=128)
        predictions = []
        for data in dataloader:
            with torch.no_grad(), autocast(device_type=self.device.type):
                batch_predictions = self.model_(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
                batch_predictions = apply_loss_fn(prediction=batch_predictions, d_output=data.d_output)
                predictions.append(batch_predictions)
        predictions = concat_predictions(predictions)
        return predictions

    def vprint(self, s: str):
        if self.verbose:
            print(s)

    def score(self, X, y) -> float:
        y_pred = self._infer(X)
        y_true = self.preprocessor_.transform_target(y)
        metric = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.preprocessor_.d_output)
        return metric



class TabSTARClassifier(BaseTabSTAR, BaseEstimator, ClassifierMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        predictions = self._infer(X)
        if predictions.ndim == 1:
            return np.round(predictions)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        return self._infer(X)

    @property
    def is_cls(self) -> bool:
        return True


class TabSTARRegressor(BaseTabSTAR, BaseEstimator, RegressorMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        z_scores = self._infer(X)
        y_pred = self.preprocessor_.inverse_transform_target(z_scores)
        return y_pred

    @property
    def is_cls(self) -> bool:
        return False


