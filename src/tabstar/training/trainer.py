import gc
import shutil
import time
from typing import Tuple, Optional

import numpy as np
import torch
from peft import PeftModel
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.tabstar_verbalizer import TabSTARData
from tabstar.training.checkpoint_averaging import CheckpointManager
from tabstar.training.dataloader import get_dataloader
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.hyperparams import set_accumulation_steps
from tabstar.training.lora import load_pretrained, load_finetuned
from tabstar.training.metrics import calculate_metric, apply_loss_fn, calculate_loss
from tabstar.training.optimizer import get_optimizer, get_scheduler
from tabstar.training.utils import concat_predictions


class TabStarTrainer:

    def __init__(self, max_epochs: int, lora_lr: float, lora_wd: float, lora_r: int, lora_alpha: float,
                 lora_dropout: float, lora_batch: int, patience: int, global_batch: int, device: torch.device,
                 model_version: str, cp_average: bool, time_limit: int, output_dir: Optional[str], val_batch_size: int):
        self.lora_batch = lora_batch
        self.global_batch = global_batch
        self.val_batch_size = val_batch_size
        self.accumulation_steps = set_accumulation_steps(global_batch=global_batch, batch_size=lora_batch)
        self.max_epochs = max_epochs
        self.device = device
        self.cp_average = cp_average
        self.model_version = model_version
        self.model = load_pretrained(model_version=model_version, lora_r=lora_r, lora_alpha=lora_alpha,
                                     dropout=lora_dropout)
        self.model.to(self.device)
        self.optimizer = get_optimizer(model=self.model, lr=lora_lr, wd=lora_wd)
        self.scheduler = get_scheduler(optimizer=self.optimizer, max_lr=lora_lr, epochs=self.max_epochs)
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = EarlyStopping(patience=patience)
        self.cp_manager = CheckpointManager(do_average=self.cp_average, output_dir=output_dir)
        self.steps: int = 0
        self.time_limit = time_limit or 60 * 60 * 10

    def train(self, train_data: TabSTARData, val_data: TabSTARData) -> float:
        train_loader = get_dataloader(train_data, is_train=True, batch_size=self.lora_batch)
        val_loader = get_dataloader(val_data, is_train=False, batch_size=self.val_batch_size)
        start_time = time.time()
        for epoch in tqdm(range(1, self.max_epochs + 1), desc="Epochs", leave=False):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = self._evaluate_epoch(val_loader)
            emoji = self.early_stopper.update(metric=val_metric)
            print(f"Epoch {epoch} || Train {train_loss:.4f} || Val {val_loss:.4f} || Metric {val_metric:.4f} {emoji}")
            if self.early_stopper.is_best:
                self.model.save_pretrained(self.cp_manager.best_dir)
            elif self.early_stopper.should_stop:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            self.scheduler.step()
            self.cp_manager.save_checkpoint(model=self.model, epoch=epoch, val_loss=val_loss)
            if self.will_next_epoch_exceed_budget(epoch=epoch, start_time=start_time):
                break
        self.cp_manager.average_checkpoints(model=self.model, evaluator=self._evaluate_epoch, val_loader=val_loader)
        best_metric = self.cp_manager.avg_metric or self.early_stopper.metric
        return best_metric

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for data in tqdm(dataloader, desc="Batches", leave=False):
            batch_loss = self._train_batch(data)
            total_loss += batch_loss * len(data.y)
            total_samples += len(data.y)
            self.steps += 1
            if self.steps % self.accumulation_steps == 0:
                self._do_update()
        if self.steps % self.accumulation_steps != 0:
            self._do_update()
        epoch_loss = total_loss / total_samples
        return epoch_loss

    def _train_batch(self, data: TabSTARData) -> float:
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            loss, predictions = self._do_forward(data=data)
            loss_for_backward = loss / self.accumulation_steps
        if self.use_amp:
            scaled_loss = self.scaler.scale(loss_for_backward)
            scaled_loss.backward()
        else:
            loss_for_backward.backward()
        original_mean_batch_loss = loss.item()
        return original_mean_batch_loss

    def _do_forward(self, data: TabSTARData) -> Tuple[Tensor, Tensor]:
        predictions = self.model(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
        loss = calculate_loss(predictions=predictions, y=data.y, d_output=data.d_output)
        return loss, predictions

    def _do_update(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _evaluate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        y_pred = []
        y_true = []
        d_output = None

        for data in dataloader:
            d_output = data.d_output
            with torch.no_grad(), autocast(device_type=self.device.type, enabled=self.use_amp):
                batch_loss, batch_predictions = self._do_forward(data=data)
                total_loss += batch_loss * len(data.y)
                total_samples += len(data.y)
                batch_predictions = apply_loss_fn(prediction=batch_predictions, d_output=d_output)
                y_pred.append(batch_predictions)
                y_true.append(data.y)
        y_pred = concat_predictions(y_pred)
        y_true = np.concatenate(y_true)
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=d_output)
        loss = total_loss / total_samples
        loss = loss.item()
        return loss, metrics.score

    def load_model(self) -> PeftModel:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.model = load_finetuned(self.cp_manager.to_load_dir, tabstar_version=self.model_version)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def delete_model(self):
        shutil.rmtree(self.cp_manager.to_load_dir)

    def will_next_epoch_exceed_budget(self, epoch: int, start_time) -> bool:
        elapsed = round(time.time() - start_time, 1)
        avg_epoch_time = round(elapsed / epoch, 1)
        next_epoch_estimate = round(elapsed + avg_epoch_time, 1)
        if next_epoch_estimate > self.time_limit:
            print(f"⏱️ Limit exceeds next epoch: {elapsed=}, {avg_epoch_time=}, {self.time_limit=}. Stopping!")
            return True
        return False
