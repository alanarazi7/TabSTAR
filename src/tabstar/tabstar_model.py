from typing import Optional, Tuple, Union

import joblib
import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import softmax

from tabstar.datasets.benchmark_folds import get_tabstar_version
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.hyperparams import LORA_LR, LORA_R, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_BATCH, GLOBAL_BATCH
from tabstar.training.metrics import calculate_metric, Metrics
from tabstar.training.trainer import TabStarTrainer
from tabstar.training.utils import concat_predictions, fix_seed


class BaseTabSTAR:
    def __init__(self,
                 lora_lr: float = LORA_LR,
                 lora_r: int = LORA_R,
                 lora_batch: int = LORA_BATCH,
                 global_batch: int = GLOBAL_BATCH,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = FINETUNE_PATIENCE,
                 verbose: bool = False,
                 device: Optional[Union[str,  torch.device]] = None,
                 random_state: Optional[int] = None,
                 pretrain_dataset_or_path: Optional[str] = None,
                 debug: bool = False):
        self.lora_lr = lora_lr
        self.lora_r = lora_r
        self.lora_batch = lora_batch
        self.global_batch = global_batch
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.debug = debug
        self.preprocessor_: Optional[TabSTARVerbalizer] = None
        self.model_: Optional[PeftModel] = None
        self.random_state = random_state
        fix_seed(seed=self.random_state)
        self.device = get_device(device=device)
        print(f"🖥️ Using device: {self.device}")
        self.use_amp = bool(self.device.type == "cuda")
        self.model_version = get_tabstar_version(pretrain_dataset_or_path=pretrain_dataset_or_path)

    def fit(self, X, y):
        if self.model_ is not None:
            raise ValueError("Model is already trained. Call fit() only once.")
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        x = X.copy()
        y = y.copy()
        train_data, val_data = self._prepare_for_train(x, y)
        self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
        trainer = TabStarTrainer(lora_lr=self.lora_lr,
                                 lora_r=self.lora_r,
                                 lora_batch=self.lora_batch,
                                 global_batch=self.global_batch,
                                 max_epochs=self.max_epochs,
                                 patience=self.patience,
                                 device=self.device,
                                 model_version=self.model_version,
                                 debug=self.debug)
        trainer.train(train_data, val_data)
        self.model_ = trainer.load_model()

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
        self.model_.eval()
        data = self.preprocessor_.transform(X, y=None)
        dataloader = get_dataloader(data, is_train=False, batch_size=128)
        predictions = []
        for data in dataloader:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                batch_predictions = self.model_(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
                if self.is_cls:
                    batch_predictions = softmax(batch_predictions.to(torch.float32), dim=1)
                predictions.append(batch_predictions)
        predictions = concat_predictions(predictions)
        return predictions

    def vprint(self, s: str):
        if self.verbose:
            print(s)

    def score(self, X, y) -> float:
        metrics = self.score_all_metrics(X=X, y=y)
        return metrics.score

    def score_all_metrics(self, X, y) -> Metrics:
        x = X.copy()
        y = y.copy()
        y_pred = self._infer(x)
        y_true = self.preprocessor_.transform_target(y)
        return calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.preprocessor_.d_output)



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

    @property
    def classes_(self) -> np.ndarray:
        if self.preprocessor_ is None or self.preprocessor_.y_values is None:
            raise ValueError("Model is not trained yet! Call fit() before accessing classes_.")
        return np.array(self.preprocessor_.y_values)


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


