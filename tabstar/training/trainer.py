from typing import Dict

import numpy as np
import torch
import wandb
from pandas import DataFrame, Series
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstar.arch.config import TabStarConfig
from tabstar.training.dataloader import get_dataloader
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.lora import load_model_with_lora
from tabstar.training.loss import LossAccumulator
from tabstar.training.optimizer import get_optimizer, get_scheduler, MAX_EPOCHS

torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


# TODO: replace with HF built in Trainer, exclude custom logics
class TabStarTrainer:

    def __init__(self):
        self.model = load_model_with_lora()
        self.optimizer = get_optimizer(model=self.model)
        self.scheduler = get_scheduler(optimizer=self.optimizer)
        self.scaler = GradScaler()
        self.early_stopper = EarlyStopping()
        # TODO the config should be initialized earlier
        self.steps: int = 0
        self.config = TabStarConfig()

    def train(self, x: DataFrame, y: Series):
        with tqdm(MAX_EPOCHS, desc="Epochs", leave=False) as pbar_epochs:
            for epoch in range(1, MAX_EPOCHS + 1):
                dataloader = get_dataloader(x=x, y=y)
                train_loss = LossAccumulator()
                with tqdm(total=len(dataloader), desc="Batches", leave=False) as pbar_batches:
                    for x, y in dataloader:
                        batch_loss = self.train_one_batch(x_cat=x_txt, x_num=x_num, y=y, properties=properties)
                        train_loss.update_batch(loss=batch_loss, n=len(y))
                        dataset2losses[properties.sid].update_batch(batch_loss=batch_loss, batch=x_txt)
                        self.steps += 1
                        if self.steps % self.config.accumulation_steps == 0:
                            self.do_update()
                        pbar_batches.update(1)
                    # If the total number of batches isn't divisible by accumulation_steps, update one last time.
                    if self.steps % self.config.accumulation_steps != 0:
                        self.do_update()
                dev_loss = LossAccumulator()
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.data_loaders[DataSplit.DEV]:
                        assert isinstance(data_loader, DataLoader) and isinstance(data_loader.dataset, HDF5Dataset)
                        properties = data_loader.dataset.properties
                        data_dev_loss, predictions = self.eval_dataset(data_loader=data_loader)
                        dev_loss += data_dev_loss
                        dev_metrics.append(predictions.score)
                        pbar_eval.update(1)
                metric_score = float(np.mean(dev_metrics))
                summarize_epoch(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss, metric_score=metric_score,
                                early_stopper=self.early_stopper, is_pretrain=self.is_pretrain)
                self.early_stopper.update(metric_score)
                if self.early_stopper.is_best:
                    self.model.save_pretrained(self.model_path)
                elif self.early_stopper.should_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                self.scheduler.step()
                pbar_epochs.update(1)
        return self.early_stopper.metric

    # @property
    # def model_path(self) -> str:
    #     return get_model_path(self.run_name, is_pretrain=self.is_pretrain)

    # def infer(self, x_txt: np.ndarray, x_num: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
    #     y_pred = self.model(x_txt=x_txt, x_num=x_num, sid=properties.sid, d_output=properties.d_effective_output)
    #     return InferenceOutput(y_pred=y_pred)
    #
    # def prepare_dev_test_dataloaders(self):
    #     for split in [DataSplit.DEV, DataSplit.TEST]:
    #         if self.is_pretrain and split == DataSplit.TEST:
    #             continue
    #         split_dirs = []
    #         for d in self.data_dirs:
    #             data = get_dataloader(data_dir=d, split=split, batch_size=self.config.batch_size)
    #             split_dirs.append(data)
    #         self.data_loaders[split] = split_dirs
    #
    # def do_forward(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
    #     inference = self.infer(x_txt=x_txt, x_num=x_num, properties=properties)
    #     loss_fn = get_loss_fn(properties.task_type)
    #     dtype = get_torch_dtype(properties.task_type)
    #     y = torch.tensor(y, dtype=dtype).to(self.device)
    #     loss = loss_fn(inference.y_pred, y)
    #     inference.loss = loss
    #     return inference
    #
    # def train_one_batch(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties) -> Loss:
    #     self.model.train()
    #     with autocast(device_type=self.device.type):
    #         inference = self.do_forward(x_txt=x_cat, x_num=x_num, y=y, properties=properties)
    #         # Divide the loss to scale gradients appropriately.
    #         loss = inference.loss / self.config.accumulation_steps
    #     verbose_print(f"Scaling the loss {loss.item():.3f} for {properties.sid} for mixed precision stability")
    #     scaled_loss = self.scaler.scale(loss)
    #     verbose_print(f"Backwarding a scaled loss of {scaled_loss:.3f}")
    #     scaled_loss.backward()
    #     return inference.to_loss
    #
    # def eval_dataset(self, data_loader: DataLoader) -> Tuple[LossAccumulator, Predictions]:
    #     self.model.eval()
    #     dev_dataset_loss = LossAccumulator()
    #     cache = PredictionsCache()
    #     properties = None
    #     for x_txt, x_num, y, properties in data_loader:
    #         assert isinstance(properties, DatasetProperties)
    #         verbose_print(f"Evaluating a batch of {properties.sid}, {len(x_txt)} examples")
    #         batch_loss = self.eval_one_batch(x_txt=x_txt, x_num=x_num, y=y, properties=properties, cache=cache)
    #         dev_dataset_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
    #     metric_score = calculate_metric(task_type=properties.task_type, y_true=cache.y_true, y_pred=cache.y_pred)
    #     predictions = Predictions(score=float(metric_score), predictions=cache.y_pred, labels=cache.y_true)
    #     return dev_dataset_loss, predictions
    #
    # def eval_one_batch(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties, cache: PredictionsCache) -> Loss:
    #     self.model.eval()
    #     with torch.no_grad(), autocast(device_type=self.device.type):
    #         inference = self.do_forward(x_txt=x_txt, x_num=x_num, y=y, properties=properties)
    #     predictions = apply_loss_fn(inference.y_pred, properties.task_type)
    #     cache.append(y=y, predictions=predictions)
    #     return inference.to_loss
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
    #
    # def do_update(self):
    #     verbose_print(f"🔄 Updating loss!")
    #     self.scaler.unscale_(self.optimizer)
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.optimizer.zero_grad()
