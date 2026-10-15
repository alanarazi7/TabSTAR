from typing import Optional, Tuple

import joblib
import numpy as np
import torch
from pandas import Series, DataFrame
from peft import PeftModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from tabstar.tabstarv1p1.tabstar_checkpoints import get_tabstar_version
from tabstar.tabstarv1p1.finetune.hyperparams import LORA_LR, LORA_BATCH, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_DROPOUT, \
    LORA_GLOBAL_BATCH, LORA_WD, LORA_R
from tabstar.tabstarv1p1.finetune.trainer import TabSTARv1p1Trainer
from tabstar.tabstarv1p1.arch.model import TabSTARModel
from tabstar.tabstarv1p1.preprocessing.tabstar_verbalizer import TabSTARv1p1Verbalizer
from tabstar.tabstarv1p1.training.loss import process_predictions, calculate_mixed_metric
from tabstar.tabstarv1p1.training.batching import TabSTARBatch
from tabstar.tabstarv1p1.finetune.dataloader import get_dataloader
from tabstar.tabstarv1p1.preprocessing.nulls import raise_if_null_target
from tabstar.tabstarv1p1.preprocessing.splits import split_to_val
from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData
from tabstar.tabstarv1p1.training.devices import get_device
from tabstar.tabstarv1p1.training.utils import fix_seed


class BaseTabSTARv1p1:
    def __init__(self,
                 lora_lr: float = LORA_LR,
                 lora_r: int = LORA_R,
                 lora_wd: float = LORA_WD,
                 lora_batch: int = LORA_BATCH,
                 lora_global_batch: int = LORA_GLOBAL_BATCH,
                 lora_dropout: float = LORA_DROPOUT,
                 max_epochs: int = MAX_EPOCHS,
                 patience: int = FINETUNE_PATIENCE,
                 verbose: bool = False,
                 device: Optional[str | torch.device] = None,
                 random_state: Optional[int] = None,
                 pretrain_dataset_or_path: Optional[str] = None,
                 preloaded_base_model: Optional[TabSTARModel] = None):
        self.preloaded_base_model = preloaded_base_model
        self.lora_lr = lora_lr
        self.lora_wd = lora_wd
        self.lora_r = lora_r
        self.lora_batch = lora_batch
        self.lora_global_batch = lora_global_batch
        self.lora_dropout = lora_dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.preprocessor_: Optional[TabSTARv1p1Verbalizer] = None
        self.model_: Optional[PeftModel] = None
        self.random_state = random_state
        fix_seed(seed=self.random_state)
        self.device = get_device(device=device)
        print(f"🖥️ Using device: {self.device}")
        self.use_amp = bool(self.device.type == "cuda")
        self.model_version = get_tabstar_version(pretrain_dataset_or_path=pretrain_dataset_or_path, is_cls=self.is_cls)
        self.save_dir: Optional[str] = None

    def fit(self, X, y) -> float:
        if self.model_ is not None:
            raise ValueError("Model is already trained. Call fit() only once.")
        self.vprint(f"Fitting model on data with shapes: X={X.shape}, y={y.shape}")
        x = X.copy()
        y = y.copy()
        trainer = TabSTARv1p1Trainer(lora_lr=self.lora_lr,
                                  lora_r=self.lora_r,
                                  lora_wd=self.lora_wd,
                                  lora_batch=self.lora_batch,
                                  lora_dropout=self.lora_dropout,
                                  lora_global_batch=self.lora_global_batch,
                                  max_epochs=self.max_epochs,
                                  patience=self.patience,
                                  is_cls=self.is_cls,
                                  device=self.device,
                                  model_version=self.model_version,
                                  preloaded_base_model=self.preloaded_base_model,
                                  random_state=self.random_state)
        try:
            train_data, val_data = self._prepare_for_train(x, y, d_num=trainer.model.base_model.d_num)
            self.vprint(f"We have: {len(train_data)} training and {len(val_data)} validation samples.")
            best_val_loss = trainer.train(train_data, val_data)
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
        return best_val_loss

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

    def _prepare_for_train(self, X, y, d_num: int) -> Tuple[TabSTARData, TabSTARData]:
        if not isinstance(X, DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not isinstance(y, Series):
            raise ValueError("y must be a pandas Series.")
        raise_if_null_target(y)
        self.vprint(f"Preparing data for training. X shape: {X.shape}, y shape: {y.shape}")
        x_train, x_val, y_train, y_val = split_to_val(x=X, y=y, is_cls=self.is_cls)
        self.vprint(f"Split to validation set. Train has {len(x_train)} samples, validation has {len(x_val)} samples.")
        if self.preprocessor_ is None:
            self.preprocessor_ = TabSTARv1p1Verbalizer(is_cls=self.is_cls, verbose=self.verbose, d_num=d_num)
            self.preprocessor_.fit(x_train, y_train)
        train_data = self.preprocessor_.transform(x_train, y_train)
        self.vprint(f"Transformed training data: {train_data.x_txt.shape=}, x_num shape: {train_data.x_num.shape=}")
        val_data = self.preprocessor_.transform(x_val, y_val)
        return train_data, val_data

    def _infer(self, X) -> np.ndarray:
        self.model_.eval()
        data = self.preprocessor_.transform(X, y=None)
        dataloader = get_dataloader(data, is_train=False, batch_size=256)
        predictions = []
        for data in dataloader:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                assert isinstance(data, TabSTARBatch)
                data.to_device(self.device, dtype=torch.bfloat16)
                batch_predictions = self.model_(data)
                bin_centers = self.model_.base_model.model.reg_head.bin_centers
                batch_predictions = process_predictions(prediction=batch_predictions, d_outputs=data.d_outputs,
                                                         bin_centers=bin_centers)
                # A binary/regression batch of exactly 1 example can come back 0-dimensional
                # (torch.cat below requires every batch at the same rank).
                batch_predictions = torch.atleast_1d(batch_predictions)
                predictions.append(batch_predictions)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.cpu().numpy()
        return predictions

    def vprint(self, s: str):
        if self.verbose:
            print(s)

    def score(self, X: DataFrame, y: Series, is_cls: bool) -> float:
        y_pred = self._infer(X)
        if is_cls:
            y = self.preprocessor_.transform_target(y)
        else:
            # Regression problems must be evaluated in their original scale
            y_pred = self.preprocessor_.inverse_transform_target(y_pred.reshape(-1, 1))
        metric = calculate_mixed_metric(y_true=y, y_pred=y_pred, d_outputs=[self.preprocessor_.d_output])
        return metric



class TabSTARv1p1Classifier(BaseTabSTARv1p1, BaseEstimator, ClassifierMixin):

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


class TabSTARv1p1Regressor(BaseTabSTARv1p1, BaseEstimator, RegressorMixin):

    def predict(self, X):
        if not isinstance(self.model_, PeftModel):
            raise ValueError("Model is not trained yet. Call fit() before predict().")
        z_scores = self._infer(X)
        y_pred = self.preprocessor_.inverse_transform_target(z_scores.reshape(-1, 1))
        return y_pred.ravel()

    @property
    def is_cls(self) -> bool:
        return False


