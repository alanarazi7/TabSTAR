from typing import Optional, Dict, Tuple, List, Any

import numpy as np
import torch
import wandb
from torch.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar_paper.pretraining.dataloaders import get_dev_dataloader
from tabstar_paper.pretraining.datasets import create_pretrain_dataset
from tabstar_paper.pretraining.hdf5 import HDF5Dataset
from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.datasets.properties import DatasetProperties
from tabular.evaluation.loss import apply_loss_fn, get_loss_fn, LossAccumulator, get_torch_dtype
from tabular.evaluation.metrics import PredictionsCache, calculate_metric
from tabular.evaluation.predictions import Predictions
from tabular.preprocessing.splits import DataSplit
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.nn_logger import log_general, log_dev_loss, log_dev_performance, log_train_loss, summarize_epoch
from tabular.utils.dataloaders import get_pretrain_epoch_dataloader, round_robin_batches
from tabular.utils.deep import print_model_summary
from tabular.utils.early_stopping import EarlyStopping
from tabular.evaluation.inference import InferenceOutput, Loss
from tabular.utils.optimizer import get_optimizer, MAX_EPOCHS
from tabular.utils.paths import get_model_path
from tabular.utils.utils import cprint, verbose_print, fix_seed

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
        cprint("Loaded pre-trained model and unfreezing all downstream layers for finetuning.")
        self.model = self.model.to(self.device)
        assert isinstance(self.model, Module)
        self.init_optimizer()

    def set_config(self) -> TabStarConfig:
        return TabStarConfig.create(self.args)

    def init_optimizer(self):
        self.optimizer, self.scheduler = get_optimizer(model=self.model, config=self.config)

    def infer(self, x_txt: np.ndarray, x_num: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
        y_pred = self.model(x_txt=x_txt, x_num=x_num, sid=properties.sid, d_output=properties.d_effective_output)
        return InferenceOutput(y_pred=y_pred)

    def train(self):
        print_model_summary(self.model)
        early_stopper = EarlyStopping(args=self.args)
        steps = 0
        with tqdm(total=self.max_epochs, desc="Epochs", leave=False) as pbar_epochs:
            for epoch in range(1, self.max_epochs + 1):
                log_general(scheduler=self.scheduler, steps=steps, epoch=epoch)
                dataloaders = get_pretrain_epoch_dataloader(data_dirs=self.data_dirs, batch_size=self.config.batch_size)
                num_batches = sum(len(dl) for dl in dataloaders)
                batches_generator = round_robin_batches(dataloaders)
                train_loss = LossAccumulator()
                dataset2losses: Dict[str, LossAccumulator] = {}
                with tqdm(total=num_batches, desc="Batches", leave=False) as pbar_batches:
                    for batch_idx, (x_txt, x_num, y, properties) in enumerate(batches_generator):
                        verbose_print(f"Training batch {batch_idx} over {properties.sid}")
                        batch_loss = self.train_one_batch(x_cat=x_txt, x_num=x_num, y=y, properties=properties)
                        train_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
                        if properties.sid not in dataset2losses:
                            dataset2losses[properties.sid] = LossAccumulator()
                        dataset2losses[properties.sid].update_batch(batch_loss=batch_loss, batch=x_txt)
                        steps += 1

                        # Update optimizer every 'accumulation_steps' batches.
                        if (batch_idx + 1) % self.config.accumulation_steps == 0:
                            self.do_update()

                        pbar_batches.update(1)

                    # If the total number of batches isn't divisible by accumulation_steps, update one last time.
                    if (batch_idx + 1) % self.config.accumulation_steps != 0:
                        self.do_update()

                log_train_loss(train_loss=train_loss, epoch=epoch, is_pretrain=self.is_pretrain___,
                               dataset2losses=dataset2losses)
                dev_loss = LossAccumulator()
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.dev_dataloaders:
                        assert isinstance(data_loader, DataLoader) and isinstance(data_loader.dataset, HDF5Dataset)
                        properties = data_loader.dataset.properties
                        data_dev_loss, predictions = self.eval_dataset(data_loader=data_loader, is_test_time=False)
                        dev_loss += data_dev_loss
                        dev_metrics.append(predictions.score)
                        log_dev_performance(properties=properties, is_pretrain=self.is_pretrain___, epoch=epoch,
                                            data_dev_loss=data_dev_loss, predictions=predictions)
                        pbar_eval.update(1)
                metric_score = float(np.mean(dev_metrics))
                log_dev_loss(is_pretrain=self.is_pretrain___, dev_loss=dev_loss, metric=metric_score, epoch=epoch)
                summarize_epoch(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss, metric_score=metric_score,
                                early_stopper=early_stopper, is_pretrain=self.is_pretrain___)
                early_stopper.update(metric_score)
                if early_stopper.is_best:
                    self.model.save_pretrained(self.model_path)
                elif early_stopper.should_stop:
                    cprint(f"Early stopping at epoch {epoch}")
                    break
                self.scheduler.step()
                pbar_epochs.update(1)
        wandb.log({'train_epochs': epoch})
        return early_stopper.metric

    def do_forward(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
        inference = self.infer(x_txt=x_txt, x_num=x_num, properties=properties)
        loss_fn = get_loss_fn(properties.task_type)
        dtype = get_torch_dtype(properties.task_type)
        y = torch.tensor(y, dtype=dtype).to(self.device)
        loss = loss_fn(inference.y_pred, y)
        inference.loss = loss
        return inference

    def train_one_batch(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> Loss:
        self.model.train()
        with autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
            # Divide the loss to scale gradients appropriately.
            loss = inference.loss / self.config.accumulation_steps
        verbose_print(f"Scaling the loss {loss.item():.3f} for {properties.sid} for mixed precision stability")
        scaled_loss = self.scaler.scale(loss)
        verbose_print(f"Backwarding a scaled loss of {scaled_loss:.3f}")
        scaled_loss.backward()
        return inference.to_loss

    def eval_dataset(self, data_loader: DataLoader, is_test_time: bool) -> Tuple[LossAccumulator, Predictions]:
        self.model.eval()
        dev_dataset_loss = LossAccumulator()
        cache = PredictionsCache()
        properties = None
        for x_txt, x_num, y, properties in data_loader:
            assert isinstance(properties, DatasetProperties)
            verbose_print(f"Evaluating a batch of {properties.sid}, {len(x_txt)} examples")
            batch_loss = self.eval_one_batch(x_txt=x_txt, x_num=x_num, y=y, properties=properties, cache=cache)
            dev_dataset_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
        metric_score = calculate_metric(task_type=properties.task_type, y_true=cache.y_true, y_pred=cache.y_pred,
                                        is_test_time=is_test_time)
        predictions = Predictions(score=float(metric_score), predictions=cache.y_pred, labels=cache.y_true)
        return dev_dataset_loss, predictions

    def eval_one_batch(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties, cache: PredictionsCache) -> Loss:
        self.model.eval()
        with torch.no_grad(), autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_txt, x_num=x_num, y=y, properties=properties)
        predictions = apply_loss_fn(inference.y_pred, properties.task_type)
        cache.append(y=y, predictions=predictions)
        return inference.to_loss

    def do_update(self):
        verbose_print(f"🔄 Updating loss!")
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()