import os.path
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.arch.arch import TabStarModel
from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.training.devices import CPU_CORES
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.metrics import apply_loss_fn, calculate_metric, calculate_loss
from tabstar.training.optimizer import get_scheduler
from tabstar.training.utils import fix_seed, concat_predictions
from tabstar_paper.pretraining.checkpoint import save_checkpoint, load_checkpoint
from tabstar_paper.pretraining.config import TabStarConfig
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader, get_pretrain_multi_dataloader, \
    MultiDatasetEpochBatches
from tabstar_paper.pretraining.datasets import create_pretrain_dataset
from tabstar_paper.pretraining.hdf5 import HDF5Dataset, DatasetProperties
from tabstar_paper.pretraining.hyperparameters import TrainingArgs
from tabstar_paper.pretraining.optimizer import get_optimizer
from tabstar_paper.pretraining.pretrain_args import PretrainArgs
from tabstar_paper.pretraining.unfreezing import unfreeze_text_encoder

from tabular.trainers.nn_logger import log_general
from tabular.utils.deep import print_model_summary
from tabular.utils.paths import get_model_path, get_checkpoint

torch.set_num_threads(CPU_CORES)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


class TabSTARPretrainer:

    def __init__(self,
                 run_name: str,
                 dataset_ids: List[TabularDatasetID],
                 device: torch.device,
                 train_args: TrainingArgs,
                 pretrain_args: PretrainArgs):
        self.train_args = train_args
        self.run_name = run_name
        self.dataset_ids = dataset_ids
        self.device = device
        self.args = pretrain_args
        self.data_dirs: List[str] = []
        self.dev_dataloaders: List[DataLoader] = []
        self.model: Optional[Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        fix_seed()
        self.initialize_model()
        self.initialize_data_dirs()
        self.steps: int = 0
        self.epoch: int = 0
        self.early_stopper = EarlyStopping(patience=train_args.patience)

    def initialize_data_dirs(self):
        for d in tqdm(self.dataset_ids, desc="Initializing data dirs", leave=False):
            data_dir = create_pretrain_dataset(dataset_id=d)
            self.data_dirs.append(data_dir)
            dev_dataloader = get_dev_dataloader(data_dir=data_dir, batch_size=self.train_args.batch_size)
            self.dev_dataloaders.append(dev_dataloader)

    def initialize_model(self):
        config = TabStarConfig(num_layers=self.args.tabular_layers, unfreeze_layers=self.args.unfreeze_layers)
        self.model = TabStarModel(config=config)
        unfreeze_text_encoder(text_encoder=self.model.text_encoder, layers_to_unfreeze=config.unfreeze_layers)
        self.model.to(self.device)
        assert isinstance(self.model, Module)
        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = get_optimizer(model=self.model, args=self.train_args)
        self.scheduler = get_scheduler(optimizer=self.optimizer, max_lr=self.train_args.learning_rate,
                                       epochs=self.train_args.epochs)

    def train(self):
        t0 = time.time()
        print_model_summary(self.model)
        print(f"💪 Pretraining for {self.run_name} over {len(self.data_dirs)} datasets on device {self.device}.")
        if self.args.checkpoint:
            self.load_checkpoint()
        dataloader = get_pretrain_multi_dataloader(data_dirs=self.data_dirs, args=self.train_args)
        assert isinstance(dataloader.dataset, MultiDatasetEpochBatches)
        with tqdm(total=self.train_args.epochs, desc="Epochs", leave=False) as pbar_epochs:
            for epoch in range(self.epoch + 1, self.train_args.epochs + 1):
                self.epoch = epoch
                log_general(scheduler=self.scheduler, steps=self.steps, epoch=self.epoch)
                dataloader.dataset.make_batches()
                train_loss = 0
                train_examples = 0
                with tqdm(total=len(dataloader), desc="Batches", leave=False) as pbar_batches:
                    for batch_idx, (x_txt, x_num, y, properties) in enumerate(dataloader):
                        batch_loss = self.train_one_batch(x_cat=x_txt, x_num=x_num, y=y, properties=properties)
                        train_loss += batch_loss * len(y)
                        train_examples += len(y)
                        self.steps += 1
                        if (batch_idx + 1) % self.train_args.accumulation_steps == 0:
                            self.do_update()
                        pbar_batches.update(1)
                    if (batch_idx + 1) % self.train_args.accumulation_steps != 0:
                        self.do_update()
                train_loss = train_loss / train_examples
                dev_loss = 0
                dev_examples = 0
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.dev_dataloaders:
                        assert isinstance(data_loader.dataset, HDF5Dataset)
                        dataset_loss, dataset_metric = self.eval_dataset(data_loader=data_loader)
                        dev_loss += dataset_loss * len(data_loader.dataset)
                        dev_examples += len(data_loader.dataset)
                        dev_metrics.append(dataset_metric)
                        pbar_eval.update(1)
                dev_metric = float(np.mean(dev_metrics))
                dev_loss = dev_loss / dev_examples
                wandb.log({'train_loss': train_loss, 'val_loss': dev_loss, 'val_metric': dev_metric}, step=epoch)
                emoji = " 🥇" if dev_metric > self.early_stopper.metric else f" 😓 [{self.early_stopper.failed}]"
                elapsed = time.time() - t0
                print(f"Epoch {epoch} || Time {elapsed:.2f} || Train {train_loss:.5f} || Val {dev_loss:.5f} || Metric {dev_metric:.5f} {emoji}")
                self.early_stopper.update(dev_metric)
                if self.early_stopper.is_best:
                    self.model.save_pretrained(get_model_path(self.run_name))
                elif self.early_stopper.should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                self.scheduler.step()
                pbar_epochs.update(1)
                self.save_checkpoint()
        wandb.log({'train_epochs': epoch})
        return self.early_stopper.metric

    def do_forward(self, x_txt: np.ndarray, x_num: Tensor, y: Tensor, properties: DatasetProperties) -> Tuple[Tensor, Tensor]:
        x_num = x_num.to(self.device)
        y = y.to(self.device)
        predictions = self.model(x_txt=x_txt, x_num=x_num, d_output=properties.d_output)
        loss = calculate_loss(predictions=predictions, y=y, d_output=properties.d_output)
        return predictions, loss

    def train_one_batch(self, x_cat: np.ndarray, x_num: Tensor, y: Tensor, properties: DatasetProperties) -> float:
        self.model.train()
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            predictions, loss = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
            # Divide the loss to scale gradients appropriately.
            loss_for_backward = loss / self.train_args.accumulation_steps
        scaled_loss = self.scaler.scale(loss_for_backward)
        scaled_loss.backward()
        return loss.item()

    def eval_dataset(self, data_loader: DataLoader) -> Tuple[float, float]:
        try:
            return self._eval_dataset(data_loader)
        except ValueError as e:
            dataset = data_loader.dataset
            assert isinstance(dataset, HDF5Dataset)
            print(f"⚠️ Error evaluating dataset {dataset.properties.name}!")
            raise e

    def _eval_dataset(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        dev_loss = 0.0
        dev_examples = 0

        y_pred = []
        y_true = []
        d_output = None

        for x_txt, x_num, y, properties in data_loader:
            d_output = properties.d_output
            assert isinstance(properties, DatasetProperties)
            with torch.no_grad(), autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions, loss = self.do_forward(x_txt=x_txt, x_num=x_num, y=y, properties=properties)
            dev_loss += loss.item() * len(y)
            dev_examples += len(y)
            batch_predictions = apply_loss_fn(predictions, d_output=properties.d_output)
            y_pred.append(batch_predictions)
            y_true.append(y)
        y_pred = concat_predictions(y_pred)
        y_true = np.concatenate(y_true)
        dev_loss = dev_loss / dev_examples
        metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=d_output)
        return dev_loss, metrics.score

    def do_update(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def save_checkpoint(self):
        if self.epoch != self.max_epochs_:
            save_path = get_checkpoint(self.run_name, epoch=self.epoch)
            save_checkpoint(save_path=save_path,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=self.epoch,
                            steps=self.steps,
                            early_stopping=self.early_stopper)
        last_epoch_checkpoint = get_checkpoint(self.run_name, epoch=self.epoch-1)
        if os.path.exists(last_epoch_checkpoint):
            os.remove(last_epoch_checkpoint)

    def load_checkpoint(self):
        load_path = get_checkpoint(self.run_name, epoch=self.args.checkpoint)
        cp = load_checkpoint(load_path=load_path,
                             model=self.model,
                             optimizer=self.optimizer,
                             scheduler=self.scheduler,
                             scaler=self.scaler,
                             early_stopping=self.early_stopper)
        self.epoch = cp.epoch
        self.steps = cp.steps
        print(f"⏪ Loaded checkpoint from {load_path} at epoch {self.epoch}.")