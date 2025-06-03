from typing import Dict, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.tabstar_verbalizer import TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.lora import load_model_with_lora
from tabstar.training.metrics import calculate_metric, apply_loss_fn
from tabstar.training.optimizer import get_optimizer, get_scheduler, MAX_EPOCHS
from tabular.tabstar.params.config import TabStarConfig

torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


# TODO: replace with HF built in Trainer, exclude custom logics
class TabStarTrainer:

    def __init__(self):
        self.device = get_device()
        self.model = load_model_with_lora()
        self.model.to(self.device)
        self.optimizer = get_optimizer(model=self.model)
        self.scheduler = get_scheduler(optimizer=self.optimizer)
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stopper = EarlyStopping()
        self.steps: int = 0
        # TODO the config should be initialized earlier, and allow hyperparameters control
        self.config = TabStarConfig()

    def train(self, train_data: TabSTARData, val_data: TabSTARData) -> float:
        train_loader = get_dataloader(train_data, is_train=True)
        val_loader = get_dataloader(val_data, is_train=False)

        for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc="Epochs", leave=False):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = self._evaluate_epoch(val_loader)
            emoji = "🥇" if val_metric > self.early_stopper.metric else "😓"
            print(f"Epoch {epoch} || Train {train_loss:.4f} || Val {val_loss:.4f} || Metric {val_metric:.4f} {emoji}")
            self.early_stopper.update(val_metric)
            if self.early_stopper.is_best:
                # self.model.save_pretrained(self.model_path)
                self._save_model()
            elif self.early_stopper.should_stop:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

            self.scheduler.step()

        return self.early_stopper.metric

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for data in dataloader:
            batch_loss = self._train_batch(data)
            total_loss += batch_loss * len(data.y)
            total_samples += len(data.y)
            self.steps += 1
            if self.steps % self.config.accumulation_steps == 0:
                self._do_update()

        if self.steps % self.config.accumulation_steps != 0:
            self._do_update()

        epoch_loss = total_loss / total_samples
        return epoch_loss


    def _train_batch(self, data: TabSTARData) -> float:
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            loss, predictions = self._do_forward(data=data)
            # Divide the loss to scale gradients
            loss = loss / self.config.accumulation_steps
        if self.use_amp:
            # Scale the loss for mixed precision stability
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
        loss = loss.item()
        return loss

    def _do_forward(self, data: TabSTARData) -> Tuple[Tensor, Tensor]:
        predictions = self.model(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
        if data.d_output == 1:
            loss_fn = MSELoss()
            dtype = torch.float32
        else:
            loss_fn = CrossEntropyLoss()
            dtype = torch.long
        y = torch.tensor(data.y, dtype=dtype).to(self.device)
        loss = loss_fn(predictions, y)
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
            with torch.no_grad(), autocast(device_type=self.device.type):
                batch_loss, batch_predictions = self._do_forward(data=data)
                total_loss += batch_loss * len(data.y)
                total_samples += len(data.y)
                batch_predictions = apply_loss_fn(prediction=batch_predictions, d_output=d_output)
                y_pred.append(batch_predictions)
                y_true.append(data.y)
        y_pred = np.concatenate([p.cpu().detach().numpy() for p in y_pred])
        y_true = np.concatenate(y_true)
        metric_score = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=d_output)
        loss = total_loss / total_samples
        loss = loss.item()
        return loss, metric_score


    # @property
    # def model_path(self) -> str:
    #     return get_model_path(self.run_name, is_pretrain=self.is_pretrain)
    #
    # def load_model(self, cp_path: str):
    #     # TODO: it doesn't seem like the Lora is being loaded here. Fix for "production" release, no need for research
    #     # We probably would like to separate between the Pretrain and the Finetune code into different classes
    #     if not exists(cp_path):
    #         raise FileNotFoundError(f"Checkpoint file {cp_path} does not exist.")
    #     self.model = TabStarModel.from_pretrained(cp_path)
    #     self.model.to(self.device)
    #
    # def test(self) -> Dict[DataSplit, Predictions]:
    #     assert not self.is_pretrain
    #     self.load_model(cp_path=self.model_path)
    #     ret = {}
    #     for split in [DataSplit.DEV, DataSplit.TEST]:
    #         data_loaders = self.data_loaders[split]
    #         assert len(data_loaders) == 1, f"Testing is only for single dataset models, but got {len(data_loaders)}"
    #         loss, predictions = self.eval_dataset(data_loader=data_loaders[0])
    #         ret[split] = predictions
    #     assert isinstance(self.args, FinetuneArgs)
    #     if not self.args.keep_model:
    #         shutil.rmtree(self.model_path)
    #     return ret
