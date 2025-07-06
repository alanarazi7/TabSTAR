from typing import Optional, Tuple, List, Any

import numpy as np
import torch
import wandb
from torch.amp import autocast, GradScaler
from torch.nn import Module, CrossEntropyLoss, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.training.metrics import apply_loss_fn, calculate_metric
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader, get_pretrain_epoch_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset
from tabstar_paper.pretraining.hdf5 import HDF5Dataset, DatasetProperties

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
from tabular.utils.early_stopping import EarlyStopping
from tabular.evaluation.inference import InferenceOutput, Loss
from tabular.utils.optimizer import get_optimizer, MAX_EPOCHS
from tabular.utils.paths import get_model_path
from tabular.utils.utils import fix_seed

torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


class TabSTARPretrainer:

    def __init__(self, run_name: str, dataset_ids: List[TabularDatasetID], device: torch.device, pretrain_args: PretrainArgs):
        fix_seed()
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
        self.max_epochs = MAX_EPOCHS
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
        self.optimizer, self.scheduler = get_optimizer(model=self.model, config=self.config)

    def train(self):
        print_model_summary(self.model)
        early_stopper = EarlyStopping(args=self.args)
        steps = 0
        examples = 0
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
                        examples += len(x_txt)

                        # Update optimizer every 'accumulation_steps' batches.
                        if (batch_idx + 1) % self.config.accumulation_steps == 0:
                            self.do_update()

                        pbar_batches.update(1)

                    # If the total number of batches isn't divisible by accumulation_steps, update one last time.
                    if (batch_idx + 1) % self.config.accumulation_steps != 0:
                        self.do_update()
                wandb.log({}, step=epoch)
                dev_loss = LossAccumulator()
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.dev_dataloaders:
                        assert isinstance(data_loader, DataLoader) and isinstance(data_loader.dataset, HDF5Dataset)
                        data_dev_loss, predictions = self.eval_dataset(data_loader=data_loader)
                        dev_loss += data_dev_loss
                        dev_metrics.append(predictions.score)
                        wandb.log({f'dataset/{data_loader.dataset.properties.name}_val_loss': data_dev_loss.avg}, step=epoch)
                        print(f"Dataset: {data_loader.dataset.properties.name} || Val Loss: {data_dev_loss.avg:.4f} || Metric: {predictions.score:.4f}")
                        pbar_eval.update(1)
                metric = float(np.mean(dev_metrics))
                wandb.log({'train_loss': train_loss.avg, 'val_loss': dev_loss.avg, 'val_metric': metric}, step=epoch)
                print(f"Steps: {steps} || Examples: {examples} || Epoch: {epoch}")
                log_str = f"Epoch {epoch} || Train {train_loss.avg} || Val {dev_loss.avg} || Metric {metric:.4f}"
                if metric > early_stopper.metric:
                    log_str += " 🥇"
                else:
                    log_str += f" 😓 [{early_stopper.epochs_without_improvement}]"
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
        if properties.is_cls:
            loss_fn = CrossEntropyLoss()
            dtype =  torch.long
        else:
            loss_fn = MSELoss()
            dtype = torch.float32
        y = torch.tensor(y, dtype=dtype).to(self.device)
        loss = loss_fn(inference.y_pred, y)
        inference.loss = loss
        return inference

    def train_one_batch(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> Loss:
        self.model.train()
        log_input_stats(x_cat=x_cat, x_num=x_num, y=y)
        with autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
            # Divide the loss to scale gradients appropriately.
            loss = inference.loss / self.config.accumulation_steps
            log_if_bad_loss(loss, step=None, properties=properties)
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
        log_grad_stats(self.model, step="before_clip")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        log_grad_stats(self.model, step="after_clip")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        log_param_stats(self.model, step="after_update")
        self.optimizer.zero_grad()


import math

def is_bad_tensor(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def log_if_bad_loss(loss, step, properties):
    if is_bad_tensor(loss):
        print(f"[BAD LOSS] Step {step} | Dataset {properties.name if hasattr(properties,'name') else properties.sid}")
        print(f"Loss value: {loss}")
        assert False

def log_input_stats(x_cat, x_num, y):
    for arr, name in zip([x_cat, x_num, y], ['x_cat', 'x_num', 'y']):
        arr_t = torch.tensor(arr)
        print(f"[Batch] {name}: min={arr_t.min().item()}, max={arr_t.max().item()}, mean={arr_t.mean().item()}")
        if is_bad_tensor(arr_t):
            print(f"[BAD INPUT] {name} at batch : NaN or Inf detected.")


def log_grad_stats(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if is_bad_tensor(grad):
                print(f"[BAD GRADIENT] {name} at step {step}: NaN or Inf detected.")
            if grad.abs().max() > 1e5:
                print(f"[LARGE GRADIENT] {name} at step {step}: max {grad.abs().max()}")

def log_param_stats(model, step):
    for name, param in model.named_parameters():
        data = param.data
        if is_bad_tensor(data):
            print(f"[BAD PARAM] {name} at step {step}: NaN or Inf detected.")
        if data.abs().max() > 1e5:
            print(f"[LARGE PARAM] {name} at step {step}: max {data.abs().max()}")