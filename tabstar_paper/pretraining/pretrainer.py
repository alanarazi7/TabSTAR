from typing import Optional, Tuple, List, Any

import numpy as np
import torch
import wandb
from torch.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.metrics import apply_loss_fn, calculate_metric, calculate_loss
from tabstar.training.optimizer import get_scheduler
from tabstar.training.utils import fix_seed
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader, get_pretrain_epoch_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset
from tabstar_paper.pretraining.hdf5 import HDF5Dataset, DatasetProperties
from tabstar_paper.pretraining.hyperparameters import PRETRAIN_PATIENCE

## TODO: Stop importing from tabular repo
from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.evaluation.loss import LossAccumulator
from tabular.evaluation.metrics import PredictionsCache
from tabular.evaluation.predictions import Predictions
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.nn_logger import log_general
from tabular.utils.dataloaders import round_robin_batches
from tabular.utils.deep import print_model_summary
from tabular.evaluation.inference import InferenceOutput, Loss
from tabular.utils.optimizer import get_groups_for_optimizer
from tabular.utils.paths import get_model_path

torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


class TabSTARPretrainer:

    def __init__(self,
                 run_name: str,
                 dataset_ids: List[TabularDatasetID],
                 max_epochs: int,
                 device: torch.device,
                 pretrain_args: PretrainArgs):
        self.run_name = run_name
        self.dataset_ids = dataset_ids
        self.device = device
        self.args = pretrain_args
        self.data_dirs: List[str] = []
        self.dev_dataloaders: List[DataLoader] = []
        self.model: Optional[Any] = None
        self.config = self.set_config()
        self.model: Optional[Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.scaler = GradScaler()
        self.max_epochs = max_epochs
        self.patience = PRETRAIN_PATIENCE
        fix_seed()
        self.initialize_model()
        self.initialize_data_dirs()

    @property
    def model_path(self) -> str:
        return get_model_path(self.run_name, is_pretrain=True)


    def initialize_data_dirs(self):
        for d in tqdm(self.dataset_ids, desc="Initializing data dirs", leave=False):
            data_dir = create_pretrain_dataset(dataset_id=d)
            self.data_dirs.append(data_dir)
            dev_dataloader = get_dev_dataloader(data_dir=data_dir, batch_size=self.config.batch_size)
            self.dev_dataloaders.append(dev_dataloader)

    def initialize_model(self):
        self.model = TabStarModel(config=self.config)
        self.model.unfreeze_textual_encoder_layers()
        self.model = self.model.to(self.device)
        assert isinstance(self.model, Module)
        self.init_optimizer()

    def set_config(self) -> TabStarConfig:
        return TabStarConfig.create(self.args)

    def init_optimizer(self):
        params = get_groups_for_optimizer(model=self.model, config=self.config)
        self.optimizer = AdamW(params)
        self.scheduler = get_scheduler(optimizer=self.optimizer, max_lr=self.config.lr, epochs=self.max_epochs)

    def train(self):
        print_model_summary(self.model)
        early_stopper = EarlyStopping(patience=self.patience)
        steps = 0
        with tqdm(total=self.max_epochs, desc="Epochs", leave=False) as pbar_epochs:
            for epoch in range(1, self.max_epochs + 1):
                log_general(scheduler=self.scheduler, steps=steps, epoch=epoch)
                dataloaders = get_pretrain_epoch_dataloader(data_dirs=self.data_dirs, batch_size=self.config.batch_size)
                num_batches = sum(len(dl) for dl in dataloaders)
                batches_generator = round_robin_batches(dataloaders)
                train_loss = LossAccumulator()
                with tqdm(total=num_batches, desc="Batches", leave=False) as pbar_batches:
                    for batch_idx, (x_txt, x_num, y, properties) in enumerate(batches_generator):
                        batch_loss = self.train_one_batch(x_cat=x_txt, x_num=x_num, y=y, properties=properties)
                        train_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
                        steps += 1

                        # Update optimizer every 'accumulation_steps' batches.
                        if (batch_idx + 1) % self.config.accumulation_steps == 0:
                            self.do_update()

                        pbar_batches.update(1)

                    # If the total number of batches isn't divisible by accumulation_steps, update one last time.
                    if (batch_idx + 1) % self.config.accumulation_steps != 0:
                        self.do_update()
                dev_loss = LossAccumulator()
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.dev_dataloaders:
                        assert isinstance(data_loader, DataLoader) and isinstance(data_loader.dataset, HDF5Dataset)
                        data_dev_loss, predictions = self.eval_dataset(data_loader=data_loader)
                        dev_loss += data_dev_loss
                        dev_metrics.append(predictions.score)
                        pbar_eval.update(1)
                metric = float(np.mean(dev_metrics))
                wandb.log({'train_loss': train_loss.avg, 'val_loss': dev_loss.avg, 'val_metric': metric}, step=epoch)
                log_str = f"Epoch {epoch} || Train {train_loss.avg} || Val {dev_loss.avg} || Metric {metric:.4f}"
                if metric > early_stopper.metric:
                    log_str += " 🥇"
                else:
                    log_str += f" 😓 [{early_stopper.failed}]"
                print(log_str)
                early_stopper.update(metric)
                if early_stopper.is_best:
                    self.model.save_pretrained(self.model_path)
                elif early_stopper.should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                self.scheduler.step()
                pbar_epochs.update(1)
        wandb.log({'train_epochs': epoch})
        return early_stopper.metric

    def do_forward(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
        y_pred = self.model(x_txt=x_txt, x_num=x_num, sid=properties.name, d_output=properties.d_output)
        inference = InferenceOutput(y_pred=y_pred)
        inference.loss = calculate_loss(predictions=y_pred, y=y, d_output=properties.d_output)
        return inference

    def train_one_batch(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> Loss:
        self.model.train()
        with autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
            # Divide the loss to scale gradients appropriately.
            loss = inference.loss / self.config.accumulation_steps
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        return inference.to_loss

    def eval_dataset(self, data_loader: DataLoader) -> Tuple[LossAccumulator, Predictions]:
        self.model.eval()
        dev_dataset_loss = LossAccumulator()
        cache = PredictionsCache()
        properties = None
        for x_txt, x_num, y, properties in data_loader:
            assert isinstance(properties, DatasetProperties)
            batch_loss = self.eval_one_batch(x_txt=x_txt, x_num=x_num, y=y, properties=properties, cache=cache)
            dev_dataset_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
        metrics = calculate_metric(y_true=cache.y_true, y_pred=cache.y_pred, d_output=properties.d_output)
        predictions = Predictions(score=metrics.score, predictions=cache.y_pred, labels=cache.y_true)
        return dev_dataset_loss, predictions

    def eval_one_batch(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties, cache: PredictionsCache) -> Loss:
        self.model.eval()
        with torch.no_grad(), autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_txt, x_num=x_num, y=y, properties=properties)
        predictions = apply_loss_fn(inference.y_pred, d_output=properties.d_output)
        cache.append(y=y, predictions=predictions)
        return inference.to_loss

    def do_update(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()