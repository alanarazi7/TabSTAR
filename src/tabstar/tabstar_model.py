from typing import Optional, Tuple, Union

import joblib
import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import softmax

from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_datasets import get_tabstar_version
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.hyperparams import LORA_LR, LORA_R, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_BATCH, GLOBAL_BATCH, \
    VAL_BATCH, LORA_WD, LORA_DROPOUT, LORA_ALPHA
from tabstar.training.metrics import calculate_metric, Metrics
from tabstar.training.trainer import TabStarTrainer
from tabstar.training.utils import concat_predictions, fix_seed, download_tabstar


class BaseTabSTAR:
    def __init__(self,
                 is_paper_version: bool = False,
                 lora_lr: float = LORA_LR,
                 lora_wd: float = LORA_WD,
                 lora_r: int = LORA_R,
                 lora_alpha: int = LORA_ALPHA,
                 lora_dropout: float = LORA_DROPOUT,
                 lora_batch: int = LORA_BATCH,
                 global_batch: int = GLOBAL_BATCH,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = FINETUNE_PATIENCE,
                 verbose: bool = False,
                 device: Optional[Union[str,  torch.device]] = None,
                 random_state: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 pretrain_dataset_or_path: Optional[str] = None,
                 keep_model: bool = True,
                 output_dir: Optional[str] = None,
                 val_batch_size: int = VAL_BATCH,
                 ):
        self.cp_average = not bool(is_paper_version)
        self.lora_lr = lora_lr
        self.lora_wd = lora_wd
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_batch = lora_batch
        self.global_batch = global_batch
        self.val_batch_size = val_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.preprocessor_: Optional[TabSTARVerbalizer] = None
        self.model_: Optional[PeftModel] = None
        self.random_state = random_state
        self.time_limit = time_limit
        self.keep_model = keep_model
        self.output_dir = output_dir
        fix_seed(seed=self.random_state)
        self.device = get_device(device=device)
        print(f"🖥️ Using device: {self.device}")
        self.use_amp = bool(self.device.type == "cuda")
        self.model_version = get_tabstar_version(pretrain_dataset_or_path=pretrain_dataset_or_path)

    def fit(self, X: DataFrame, y: Series, x_val: Optional[DataFrame] = None, y_val: Optional[DataFrame] = None):
        if self.model_ is not None:
            raise ValueError("Model is already trained. Call fit() only once.")
        self.download_base_model()
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        train_data, val_data = self._prepare_for_train(X, y, x_val, y_val)
        self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
        trainer = TabStarTrainer(lora_lr=self.lora_lr,
                                 lora_wd=self.lora_wd,
                                 lora_r=self.lora_r,
                                 lora_alpha=self.lora_alpha,
                                 lora_dropout=self.lora_dropout,
                                 lora_batch=self.lora_batch,
                                 global_batch=self.global_batch,
                                 max_epochs=self.max_epochs,
                                 patience=self.patience,
                                 device=self.device,
                                 model_version=self.model_version,
                                 cp_average=self.cp_average,
                                 time_limit=self.time_limit,
                                 output_dir=self.output_dir,
                                 val_batch_size=self.val_batch_size)
        trainer.train(train_data, val_data)
        self.model_ = trainer.load_model()
        if not self.keep_model:
            trainer.delete_model()

    @staticmethod
    def download_base_model():
        download_tabstar()

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

    def _prepare_for_train(self, X, y, x_val: Optional[DataFrame], y_val: Optional[Series]) -> Tuple[TabSTARData, TabSTARData]:
        if not isinstance(X, DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(y, Series):
            raise ValueError("y must be a pandas Series.")
        raise_if_null_target(y)
        self.vprint(f"Preparing data for training. X shape: {X.shape}, y shape: {y.shape}")
        if x_val is None and y_val is None:
            x_train, x_val, y_train, y_val = split_to_val(x=X, y=y, is_cls=self.is_cls)
            self.vprint(f"Split to validation set. Train has {len(x_train)} samples, validation has {len(x_val)} samples.")
        else:
            x_train = X.copy()
            y_train = y.copy()
            raise_if_null_target(y_val)
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
        dataloader = get_dataloader(data, is_train=False, batch_size=self.val_batch_size)
        predictions = []
        for data in dataloader:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                batch_predictions = self.model_(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
            batch_predictions = batch_predictions.to(torch.float32)
            if self.is_cls:
                batch_predictions = softmax(batch_predictions, dim=1)
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
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.preprocessor_.d_output)
        return metrics



class TabSTARClassifier(BaseTabSTAR, BaseEstimator, ClassifierMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        predictions = self._infer(X)
        if predictions.ndim == 1:
            predictions = np.round(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)
        label = self.preprocessor_.target_transformer.inverse_transform(predictions)
        return label

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
        return y_pred.flatten()

    @property
    def is_cls(self) -> bool:
        return False


