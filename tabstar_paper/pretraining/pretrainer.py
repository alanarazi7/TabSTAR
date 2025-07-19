from typing import Optional, Tuple, List, Any

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.metrics import apply_loss_fn, calculate_metric, calculate_loss
from tabstar.training.optimizer import get_scheduler
from tabstar.training.utils import fix_seed, concat_predictions
from tabstar_paper.pretraining.dataloaders import get_dev_dataloader, get_pretrain_epoch_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset
from tabstar_paper.pretraining.hdf5 import HDF5Dataset, DatasetProperties
from tabstar_paper.pretraining.hyperparameters import PRETRAIN_PATIENCE

## TODO: Stop importing from tabular repo
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.nn_logger import log_general
from tabular.utils.dataloaders import round_robin_batches
from tabular.utils.deep import print_model_summary
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
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_epochs = max_epochs
        self.patience = PRETRAIN_PATIENCE
        fix_seed()
        self.initialize_model()
        self.initialize_data_dirs()

    @property
    def model_path(self) -> str:
        return get_model_path(self.run_name)


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
                train_loss = 0
                train_examples = 0
                with tqdm(total=num_batches, desc="Batches", leave=False) as pbar_batches:
                    for batch_idx, (x_txt, x_num, y, properties) in enumerate(batches_generator):
                        batch_loss = self.train_one_batch(x_cat=x_txt, x_num=x_num, y=y, properties=properties)
                        train_loss += batch_loss * len(y)
                        train_examples += len(y)
                        steps += 1
                        if (batch_idx + 1) % self.config.accumulation_steps == 0:
                            self.do_update()
                        pbar_batches.update(1)
                    if (batch_idx + 1) % self.config.accumulation_steps != 0:
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
                emoji = " 🥇" if dev_metric > early_stopper.metric else f" 😓 [{early_stopper.failed}]"
                print(f"Epoch {epoch} || Train {train_loss:.6f} || Val {dev_loss:.6f} || Metric {dev_metric:.6f} {emoji}")
                early_stopper.update(dev_metric)
                if early_stopper.is_best:
                    self.model.save_pretrained(self.model_path)
                elif early_stopper.should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                self.scheduler.step()
                pbar_epochs.update(1)
        wandb.log({'train_epochs': epoch})
        return early_stopper.metric

    def do_forward(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> Tuple[Tensor, Tensor]:
        predictions = self.model(x_txt=x_txt, x_num=x_num, sid=properties.name, d_output=properties.d_output)
        loss = calculate_loss(predictions=predictions, y=y, d_output=properties.d_output)
        return predictions, loss

    def train_one_batch(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> float:
        self.model.train()
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            predictions, loss = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
            # Divide the loss to scale gradients appropriately.
            loss_for_backward = loss / self.config.accumulation_steps
        scaled_loss = self.scaler.scale(loss_for_backward)
        scaled_loss.backward()
        return loss.item()

    def eval_dataset(self, data_loader: DataLoader) -> Tuple[float, float]:
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