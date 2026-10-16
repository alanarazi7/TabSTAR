from typing import Optional, Tuple

import joblib
import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from tabstar.tabstarv1p1.tabstar_checkpoints import get_tabstar_version, download_tabstar
from tabstar.tabstarv1p1.finetune.hyperparams import LORA_LR, LORA_BATCH, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_DROPOUT, \
    GLOBAL_BATCH, LORA_WD, LORA_R, LORA_ALPHA, VAL_BATCH
from tabstar.tabstarv1p1.finetune.trainer import TabSTARv1p1Trainer
from tabstar.tabstarv1p1.arch.model import TabSTARModel
from tabstar.tabstarv1p1.preprocessing.tabstar_verbalizer import TabSTARv1p1Verbalizer
from tabstar.tabstarv1p1.training.loss import process_predictions
from tabstar.tabstarv1p1.training.metrics import calculate_metric, Metrics
from tabstar.tabstarv1p1.training.batching import TabSTARBatch
from tabstar.tabstarv1p1.finetune.dataloader import get_dataloader
from tabstar.tabstarv1p1.preprocessing.nulls import raise_if_null_target
from tabstar.tabstarv1p1.preprocessing.splits import split_to_val
from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData
from tabstar.tabstarv1p1.training.devices import get_device, get_precision
from tabstar.tabstarv1p1.training.utils import fix_seed


class BaseTabSTARv1p1:
    def __init__(self,
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
                 device: Optional[str | torch.device] = None,
                 random_state: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 pretrain_dataset_or_path: Optional[str] = None,
                 keep_model: bool = True,
                 output_dir: Optional[str] = None,
                 val_batch_size: int = VAL_BATCH,
                 preloaded_base_model: Optional[TabSTARModel] = None,
                 ):
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
        self.preprocessor_: Optional[TabSTARv1p1Verbalizer] = None
        self.model_: Optional[PeftModel] = None
        self.random_state = random_state
        self.time_limit = time_limit
        self.pretrain_dataset_or_path = pretrain_dataset_or_path
        self.keep_model = keep_model
        self.output_dir = output_dir
        self.preloaded_base_model = preloaded_base_model
        self.best_val_loss_: Optional[float] = None
        fix_seed(seed=self.random_state)
        self.device = get_device(device=device)
        print(f"🖥️ Using device: {self.device}")
        self.use_amp = bool(self.device.type == "cuda")
        self.model_version = get_tabstar_version(pretrain_dataset_or_path=pretrain_dataset_or_path, is_cls=self.is_cls)
        self.save_dir: Optional[str] = None

    def fit(self, X: DataFrame, y: Series, x_val: Optional[DataFrame] = None, y_val: Optional[Series] = None):
        if self.model_ is not None:
            raise ValueError("Model is already trained. Call fit() only once.")
        download_tabstar(repo_id=self.model_version)
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        trainer = TabSTARv1p1Trainer(lora_lr=self.lora_lr,
                                  lora_wd=self.lora_wd,
                                  lora_r=self.lora_r,
                                  lora_alpha=self.lora_alpha,
                                  lora_dropout=self.lora_dropout,
                                  lora_batch=self.lora_batch,
                                  global_batch=self.global_batch,
                                  max_epochs=self.max_epochs,
                                  patience=self.patience,
                                  is_cls=self.is_cls,
                                  device=self.device,
                                  model_version=self.model_version,
                                  val_batch_size=self.val_batch_size,
                                  time_limit=self.time_limit,
                                  output_dir=self.output_dir,
                                  preloaded_base_model=self.preloaded_base_model,
                                  random_state=self.random_state)
        try:
            train_data, val_data = self._prepare_for_train(X, y, x_val, y_val,
                                                           d_num=trainer.model.base_model.d_num)
            self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
            self.best_val_loss_ = trainer.train(train_data, val_data)
        except Exception:
            if self.preloaded_base_model is not None:
                # A crash anywhere from here on (data prep or mid-training, e.g. a CUDA OOM under
                # GPU contention with another live process) skips the unload() that load_model()
                # would otherwise do below, leaving the caller's shared base model still
                # LoRA-wrapped (get_peft_model mutates its submodules in place). The next finetune
                # reusing it then silently corrupts: peft reports "No modules were targeted for
                # adaptation" once double-wrapped, and every subsequent dataset in a
                # do_finetunability_check.py-style loop fails the same way. Unload before
                # re-raising so one dataset's crash doesn't cascade.
                trainer.model.unload()
            raise
        self.model_ = trainer.load_model()
        self.save_dir = trainer.save_dir
        if not self.keep_model:
            trainer.delete_model()
        return self

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
    def load(cls, path: str) -> "BaseTabSTARv1p1":
        return joblib.load(path)

    def _prepare_for_train(self, X, y, x_val: Optional[DataFrame], y_val: Optional[Series],
                           d_num: int) -> Tuple[TabSTARData, TabSTARData]:
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
            self.preprocessor_ = TabSTARv1p1Verbalizer(is_cls=self.is_cls, verbose=self.verbose, d_num=d_num)
            self.preprocessor_.fit(x_train, y_train)
        train_data = self.preprocessor_.transform(x_train, y_train)
        self.vprint(f"Transformed training data: {train_data.x_txt.shape=}, x_num shape: {train_data.x_num.shape=}")
        val_data = self.preprocessor_.transform(x_val, y_val)
        return train_data, val_data

    def _infer(self, X) -> np.ndarray:
        """Mean of the predictions of every adapter the model carries, one per selected epoch."""
        self.model_.eval()
        data = self.preprocessor_.transform(X, y=None)
        adapter_predictions = []
        for adapter_name in self.model_.peft_config:
            self.model_.set_adapter(adapter_name)
            adapter_predictions.append(self._infer_adapter(data))
        return np.mean(adapter_predictions, axis=0)

    def _infer_adapter(self, data: TabSTARData) -> np.ndarray:
        dataloader = get_dataloader(data, is_train=False, batch_size=self.val_batch_size)
        predictions = []
        for data in dataloader:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                assert isinstance(data, TabSTARBatch)
                data.to_device(self.device, dtype=get_precision(device=self.device))
                batch_predictions = self.model_(data)
                bin_centers = self.model_.base_model.model.reg_head.bin_centers
                batch_predictions = process_predictions(prediction=batch_predictions, d_outputs=data.d_outputs,
                                                         bin_centers=bin_centers)
                # A binary/regression batch of exactly 1 example can come back 0-dimensional
                # (torch.cat below requires every batch at the same rank).
                batch_predictions = torch.atleast_1d(batch_predictions)
                predictions.append(batch_predictions)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.to(torch.float32).cpu().numpy()
        return predictions

    def vprint(self, s: str):
        if self.verbose:
            print(s)

    def score(self, X: DataFrame, y: Series) -> float:
        metrics = self.score_all_metrics(X=X, y=y)
        return metrics.score

    def score_all_metrics(self, X: DataFrame, y: Series) -> Metrics:
        x = X.copy()
        y = y.copy()
        y_pred = self._infer(x)
        if self.is_cls:
            y_true = self.preprocessor_.transform_target(y)
        else:
            # Regression is scored in the target's original scale.
            y_true = np.asarray(y, dtype=float)
            y_pred = self.preprocessor_.inverse_transform_target(y_pred.reshape(-1, 1)).ravel()
        return calculate_metric(y_true=y_true, y_pred=y_pred, d_output=self.preprocessor_.d_output)


class TabSTARv1p1Classifier(BaseTabSTARv1p1, BaseEstimator, ClassifierMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        predictions = self._infer(X)
        if predictions.ndim == 1:
            predictions = np.round(predictions)
        else:
            predictions = np.argmax(predictions, axis=1)
        label = self.preprocessor_.target_transformer.inverse_transform(predictions.astype(int))
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


class TabSTARv1p1Regressor(BaseTabSTARv1p1, BaseEstimator, RegressorMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        z_scores = self._infer(X)
        y_pred = self.preprocessor_.inverse_transform_target(z_scores.reshape(-1, 1))
        return y_pred.flatten()

    @property
    def is_cls(self) -> bool:
        return False
