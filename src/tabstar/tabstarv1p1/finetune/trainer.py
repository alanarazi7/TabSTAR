import gc
import shutil
import time
from math import ceil
from os.path import join
from typing import Optional, Tuple

import torch
from peft import PeftModel
from tabstar.tabstarv1p1.training.utils import fix_seed
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.tabstarv1p1.training.devices import clear_cuda_cache
from tabstar.tabstarv1p1.training.hyperparams import set_accumulation_steps
from tabstar.tabstarv1p1.training.optimizer import get_step_scheduler

from tabstar.tabstarv1p1.finetune.checkpoint import make_run_dir
from tabstar.tabstarv1p1.finetune.hyperparams import VAL_BATCH
from tabstar.tabstarv1p1.finetune.lora import load_pretrained_base, load_finetuned, wrap_with_lora
from tabstar.tabstarv1p1.arch.model import TabSTARModel
from tabstar.tabstarv1p1.finetune.optimizer import get_lora_optimizer
from tabstar.tabstarv1p1.training.loss import calculate_mixed_loss, process_predictions, calculate_mixed_metric
from tabstar.tabstarv1p1.training.batching import TabSTARBatch
from tabstar.tabstarv1p1.finetune.dataloader import get_dataloader
from tabstar.tabstarv1p1.training.early_stopping import EarlyStopping
from tabstar.tabstarv1p1.training.precision import PrecisionManager
from tabstar.tabstarv1p1.preprocessing.verbalizer import TabSTARData


class TabSTARv1p1Trainer:

    def __init__(self,
                 max_epochs: int,
                 lora_lr: float,
                 lora_wd: float,
                 lora_r: int,
                 lora_alpha: int,
                 lora_batch: int,
                 lora_dropout: float,
                 global_batch: int,
                 patience: int,
                 device: torch.device,
                 model_version: str,
                 is_cls: bool,
                 val_batch_size: int = VAL_BATCH,
                 time_limit: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 preloaded_base_model: Optional[TabSTARModel] = None,
                 to_unfreeze: int = 6,
                 random_state: Optional[int] = None):
        self.lora_lr = lora_lr
        self.lora_wd = lora_wd
        self.lora_batch = lora_batch
        self.val_batch_size = val_batch_size
        self.time_limit = time_limit or 60 * 60 * 10
        self.accumulation_steps = set_accumulation_steps(global_batch=global_batch, batch_size=lora_batch)
        print(f"👣 Using {self.accumulation_steps} steps, with batch size {lora_batch}, global of {global_batch}")
        self.max_epochs = max_epochs
        self.device = device
        self.model_version = model_version
        if preloaded_base_model is not None:
            # Caller already loaded (and owns the lifecycle of) the frozen base — e.g. finetuning
            # on many datasets from the same checkpoint in one process. Skip the disk load;
            # wrap_with_lora leaves the base's own weights untouched (LoRA freezes them), so the
            # caller can `.unload()` the returned PeftModel and reuse the same base again.
            self.model = wrap_with_lora(preloaded_base_model, lora_r=lora_r, lora_alpha=lora_alpha,
                                       dropout=lora_dropout, to_unfreeze=to_unfreeze, is_cls=is_cls)
        else:
            self.model = wrap_with_lora(load_pretrained_base(model_version), lora_r=lora_r, lora_alpha=lora_alpha,
                                       dropout=lora_dropout, to_unfreeze=to_unfreeze, is_cls=is_cls)
        self.model.to(self.device)
        self.precision_manager = PrecisionManager(device=self.device)
        fix_seed(seed=random_state)
        self.optimizer = get_lora_optimizer(model=self.model, lr=self.lora_lr, wd=self.lora_wd)
        # Built in train(), where the number of optimizer updates per epoch is known.
        self.scheduler: Optional[LRScheduler] = None
        self.early_stopper = EarlyStopping(patience=patience)
        self.save_dir = make_run_dir(output_dir=output_dir)
        self.best_dir = join(self.save_dir, "best_model")
        self.steps: int = 0

    @property
    def base_model(self) -> TabSTARModel:
        return self.model.base_model.model

    def train(self, train_data: TabSTARData, val_data: TabSTARData) -> float:
        train_loader = get_dataloader(train_data, is_train=True, batch_size=self.lora_batch)
        val_loader = get_dataloader(val_data, is_train=False, batch_size=self.val_batch_size)
        steps_per_epoch = ceil(len(train_loader) / self.accumulation_steps)
        self.scheduler = get_step_scheduler(optimizer=self.optimizer, max_lr=self.lora_lr,
                                            steps_per_epoch=steps_per_epoch, epochs=self.max_epochs)
        start_time = time.time()
        for epoch in tqdm(range(1, self.max_epochs + 1), desc="Epochs", leave=False):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = self._evaluate_epoch(val_loader)
            emoji = self.early_stopper.update_loss(loss=val_loss)
            print(f"Epoch {epoch} || Train {train_loss:.4f} || Val {val_loss:.4f} || Metric {val_metric:.4f} {emoji}")
            if self.early_stopper.is_best:
                self.model.save_pretrained(self.best_dir)
            elif self.early_stopper.should_stop:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            if self.will_next_epoch_exceed_budget(epoch=epoch, start_time=start_time):
                break
        return self.early_stopper.loss

    def will_next_epoch_exceed_budget(self, epoch: int, start_time: float) -> bool:
        elapsed = round(time.time() - start_time, 1)
        avg_epoch_time = round(elapsed / epoch, 1)
        next_epoch_estimate = round(elapsed + avg_epoch_time, 1)
        if next_epoch_estimate > self.time_limit:
            print(f"⏱️ Limit exceeds next epoch: {elapsed=}, {avg_epoch_time=}, {self.time_limit=}. Stopping!")
            return True
        return False

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
                self.precision_manager.update_model(model=self.model, optimizer=self.optimizer)
                self.scheduler.step()
        if self.steps % self.accumulation_steps != 0:
            self.precision_manager.update_model(model=self.model, optimizer=self.optimizer)
            self.scheduler.step()
        epoch_loss = total_loss / total_samples
        return epoch_loss

    def _train_batch(self, data: TabSTARBatch) -> float:
        assert isinstance(data, TabSTARBatch)
        return self._train_batch_with_oom_retry(data, weight=1.0)

    def _train_batch_with_oom_retry(self, data: TabSTARBatch, weight: float) -> float:
        """Forward and backward OOM retries by splitting the batch in half and recursing, each
        half's loss weighted by its share of the original batch so the accumulated gradient
        matches what a single, un-split backward pass over the full batch would have produced.
        A backward call that OOMs partway through may already have applied gradients for part of
        the graph, so on that failure the current accumulation window's gradients are wiped via
        zero_grad() before retrying, rather than assuming which parameters got a partial update.
        """
        try:
            loss, predictions = self._do_forward(data=data)
        except torch.cuda.OutOfMemoryError:
            return self._retry_split(data, weight)
        loss_for_backward = loss * weight / self.accumulation_steps
        try:
            self.precision_manager.backward_loss(loss_for_backward=loss_for_backward)
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            return self._retry_split(data, weight)
        return loss.item()

    def _retry_split(self, data: TabSTARBatch, weight: float) -> float:
        batch_size = len(data.y)
        if batch_size == 1:
            raise torch.cuda.OutOfMemoryError
        clear_cuda_cache()
        mid = batch_size // 2
        remainder = batch_size - mid
        print(f"🤯 Splitting training batch of {batch_size} in half due to OOM")
        first_weight = weight * mid / batch_size
        second_weight = weight * remainder / batch_size
        first_loss = self._train_batch_with_oom_retry(data.slice(0, mid), weight=first_weight)
        second_loss = self._train_batch_with_oom_retry(data.slice(mid, batch_size), weight=second_weight)
        return (first_loss * mid + second_loss * remainder) / batch_size

    def _do_forward(self, data: TabSTARBatch) -> Tuple[Tensor, Tensor]:
        data.to_device(self.device, dtype=self.precision_manager.dtype)
        assert isinstance(data, TabSTARBatch)
        with self.precision_manager.autocast_context():
            predictions = self.model(data)
        base_model = self.base_model
        loss = calculate_mixed_loss(predictions=predictions, y=data.y, d_outputs=data.d_outputs,
                                    bin_edges=base_model.reg_head.bin_edges,
                                    bin_centers=base_model.reg_head.bin_centers)
        return loss, predictions

    def _evaluate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        y_pred = []
        y_true = []
        d_output = None

        for data in dataloader:
            d_output = data.d_outputs
            with torch.no_grad():
                batch_loss, batch_predictions = self._do_forward(data=data)
            total_loss += batch_loss * len(data.y)
            total_samples += len(data.y)
            bin_centers = self.base_model.reg_head.bin_centers
            batch_predictions = process_predictions(prediction=batch_predictions, d_outputs=d_output,
                                                     bin_centers=bin_centers)
            y_pred.append(batch_predictions)
            y_true.append(data.y)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        metric = calculate_mixed_metric(y_true=y_true, y_pred=y_pred, d_outputs=d_output)
        loss = total_loss / total_samples
        loss = loss.item()
        return loss, metric

    def load_model(self) -> PeftModel:
        # Strip the LoRA layers before dropping this reference: with a caller-owned
        # preloaded_base_model, get_peft_model wrapped it in place, and the caller needs it back
        # clean (no LoRA modules attached) to reuse for the next dataset's finetune.
        self.model.unload()
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.model = load_finetuned(self.best_dir, tabstar_version=self.model_version)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def delete_model(self):
        shutil.rmtree(self.save_dir)
