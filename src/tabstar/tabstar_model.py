from typing import Optional, Tuple

import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import autocast

from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.metrics import apply_loss_fn
from tabstar.training.trainer import TabStarTrainer
from tabstar.training.utils import concat_predictions


class BaseTabSTAR:
    def __init__(self, verbose: bool = False, preprocessor: Optional[TabSTARVerbalizer] = None):
        self.verbose = verbose
        self.preprocessor_ = preprocessor
        self.model_: Optional[PeftModel] = None
        self.device = get_device()

    def fit(self, X, y):
        train_data, val_data = self._prepare_for_train(X, y)
        trainer = TabStarTrainer()
        trainer.train(train_data, val_data)
        trainer.load_model()
        self.model_ = trainer.model

    def predict(self, X):
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def is_cls(self) -> bool:
        raise NotImplementedError("Must be implemented in subclass")

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


