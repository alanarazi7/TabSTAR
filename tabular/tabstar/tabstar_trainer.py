import shutil
from os.path import exists
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import wandb
from peft import LoraConfig, get_peft_model
from torch.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.datasets.properties import DatasetProperties
from tabular.datasets.torch_dataset import HDF5Dataset
from tabular.evaluation.loss import apply_loss_fn, get_loss_fn, LossAccumulator, get_torch_dtype
from tabular.evaluation.metrics import PredictionsCache, calculate_metric
from tabular.evaluation.predictions import Predictions
from tabular.models.abstract_model import TabularModel
from tabular.preprocessing.objects import PreprocessingMethod
from tabular.preprocessing.splits import DataSplit
from tabular.tabstar.arch.arch import TabStarModel
from tabular.tabstar.params.config import TabStarConfig
from tabular.tabstar.params.constants import E5_LAYERS
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.trainers.nn_logger import log_general, log_dev_loss, log_dev_performance, log_train_loss, summarize_epoch
from tabular.utils.dataloaders import get_pretrain_epoch_dataloader, get_dataloader, round_robin_batches
from tabular.utils.deep import print_model_summary, get_last_layers_num
from tabular.utils.early_stopping import EarlyStopping
from tabular.evaluation.inference import InferenceOutput, Loss
from tabular.utils.optimizer import get_optimizer, MAX_EPOCHS
from tabular.utils.paths import get_model_path
from tabular.utils.utils import cprint, verbose_print, fix_seed

torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


# TODO: replace all this with HF built in Trainer, exclude custom logics
class TabStarTrainer(TabularModel):

    MODEL_NAME = "TabSTAR 🌟"
    SHORT_NAME = "Tab*"
    PROCESSING = PreprocessingMethod.TABSTAR

    def __init__(self, run_name: str, dataset_ids: List[TabularDatasetID], device: torch.device, run_num: int = 0,
                 train_examples: int = 0, args: Optional[PretrainArgs] = None,
                 carte_lr_index: Optional[int] = None):
        super().__init__(run_name=run_name, dataset_ids=dataset_ids, device=device, run_num=run_num,
                         train_examples=train_examples, args=args, carte_lr_index=carte_lr_index)
        cprint(f"Initialized the network {self.MODEL_NAME}")
        self.is_pretrain = isinstance(self.args, PretrainArgs)
        self.model: Optional[Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.scaler = GradScaler()
        self.max_epochs = MAX_EPOCHS
        self.data_loaders: Dict[DataSplit, List[DataLoader]] = {}
        fix_seed()

    @property
    def model_path(self) -> str:
        return get_model_path(self.run_name, is_pretrain=self.is_pretrain)

    def initialize_model(self):
        if self.is_pretrain:
            # For pretraining, create a new model and unfreeze textual encoder layers.
            self.model = TabStarModel(config=self.config)
            self.model.unfreeze_textual_encoder_layers()
            cprint("Loaded pre-trained model and unfreezing all downstream layers for finetuning.")
        else:
            # For finetuning, load a pre-trained model from the checkpoint and wrap with Lora.
            assert isinstance(self.args, FinetuneArgs)
            pretrain_path = get_model_path(self.args.pretrain_args.full_exp_name, is_pretrain=True)
            self.model = TabStarModel.from_pretrained(pretrain_path)
            self.model.config = self.config
            self.wrap_with_lora()
        self.model = self.model.to(self.device)
        assert isinstance(self.model, Module)
        self.init_optimizer()
        dataset_names = '\n🗂️ '.join([''] + [str(d) for d in self.datasets])
        cprint(f"Initializing model {self.MODEL_NAME} for datasets:{dataset_names}")

    def set_config(self) -> TabStarConfig:
        return TabStarConfig.create(self.args)

    def init_optimizer(self):
        self.optimizer, self.scheduler = get_optimizer(model=self.model, config=self.config)

    def infer(self, x_txt: np.ndarray, x_num: np.ndarray, properties: DatasetProperties) -> InferenceOutput:
        y_pred = self.model(x_txt=x_txt, x_num=x_num, sid=properties.sid, d_output=properties.d_effective_output)
        return InferenceOutput(y_pred=y_pred)

    def train(self):
        print_model_summary(self.model)
        if not self.dataset_ids:
            cprint(f"No datasets to train on for {self.MODEL_NAME}")
            self.model.save_pretrained(self.model_path)
            return 0.0
        cprint(f"Training {self.MODEL_NAME}!")
        self.prepare_dev_test_dataloaders()
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

                log_train_loss(train_loss=train_loss, epoch=epoch, is_pretrain=self.is_pretrain,
                               dataset2losses=dataset2losses)
                dev_loss = LossAccumulator()
                dev_metrics = []
                with tqdm(total=len(self.data_dirs), desc="Eval", leave=False) as pbar_eval:
                    for data_loader in self.data_loaders[DataSplit.DEV]:
                        assert isinstance(data_loader, DataLoader) and isinstance(data_loader.dataset, HDF5Dataset)
                        properties = data_loader.dataset.properties
                        data_dev_loss, predictions = self.eval_dataset(data_loader=data_loader)
                        dev_loss += data_dev_loss
                        dev_metrics.append(predictions.score)
                        log_dev_performance(properties=properties, is_pretrain=self.is_pretrain, epoch=epoch,
                                            data_dev_loss=data_dev_loss, predictions=predictions)
                        pbar_eval.update(1)
                metric_score = float(np.mean(dev_metrics))
                log_dev_loss(is_pretrain=self.is_pretrain, dev_loss=dev_loss, metric=metric_score, epoch=epoch)
                summarize_epoch(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss, metric_score=metric_score,
                                early_stopper=early_stopper, is_pretrain=self.is_pretrain)
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

    def prepare_dev_test_dataloaders(self):
        for split in [DataSplit.DEV, DataSplit.TEST]:
            if self.is_pretrain and split == DataSplit.TEST:
                continue
            split_dirs = []
            for d in self.data_dirs:
                data = get_dataloader(data_dir=d, split=split, batch_size=self.config.batch_size)
                split_dirs.append(data)
            self.data_loaders[split] = split_dirs

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

    def eval_dataset(self, data_loader: DataLoader) -> Tuple[LossAccumulator, Predictions]:
        self.model.eval()
        dev_dataset_loss = LossAccumulator()
        cache = PredictionsCache()
        properties = None
        for x_txt, x_num, y, properties in data_loader:
            assert isinstance(properties, DatasetProperties)
            verbose_print(f"Evaluating a batch of {properties.sid}, {len(x_txt)} examples")
            batch_loss = self.eval_one_batch(x_txt=x_txt, x_num=x_num, y=y, properties=properties, cache=cache)
            dev_dataset_loss.update_batch(batch_loss=batch_loss, batch=x_txt)
        metric_score = calculate_metric(task_type=properties.task_type, y_true=cache.y_true, y_pred=cache.y_pred)
        predictions = Predictions(score=float(metric_score), predictions=cache.y_pred, labels=cache.y_true)
        return dev_dataset_loss, predictions

    def eval_one_batch(self, x_txt: np.ndarray, x_num: np.ndarray, y: np.ndarray, properties: DatasetProperties, cache: PredictionsCache) -> Loss:
        self.model.eval()
        with torch.no_grad(), autocast(device_type=self.device.type):
            inference = self.do_forward(x_txt=x_txt, x_num=x_num, y=y, properties=properties)
        predictions = apply_loss_fn(inference.y_pred, properties.task_type)
        cache.append(y=y, predictions=predictions)
        return inference.to_loss

    def load_model(self, cp_path: str):
        # TODO: it doesn't seem like the Lora is being loaded here. Fix for "production" release, no need for research
        # We probably would like to separate between the Pretrain and the Finetune code into different classes
        if not exists(cp_path):
            raise FileNotFoundError(f"Checkpoint file {cp_path} does not exist.")
        self.model = TabStarModel.from_pretrained(cp_path)
        self.model.to(self.device)

    def test(self) -> Dict[DataSplit, Predictions]:
        assert not self.is_pretrain
        self.load_model(cp_path=self.model_path)
        ret = {}
        for split in [DataSplit.DEV, DataSplit.TEST]:
            data_loaders = self.data_loaders[split]
            assert len(data_loaders) == 1, f"Testing is only for single dataset models, but got {len(data_loaders)}"
            loss, predictions = self.eval_dataset(data_loader=data_loaders[0])
            ret[split] = predictions
        assert isinstance(self.args, FinetuneArgs)
        if not self.args.keep_model:
            shutil.rmtree(self.model_path)
        return ret

    def do_update(self):
        verbose_print(f"🔄 Updating loss!")
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def wrap_with_lora(self):
        # TODO: probably best if this is written more generic and not so hard-coded
        lora_modules = ["query", "key", "value", "out_proj", "linear1", "linear2",
                        "cls_head.layers.0", "reg_head.layers.0"]
        to_unfreeze = get_last_layers_num(to_unfreeze=self.config.unfreeze_layers)
        to_freeze = [i for i in range(E5_LAYERS) if i not in to_unfreeze]
        to_exclude = []
        for i in to_freeze:
            for name, _ in self.model.named_modules():
                if name.startswith(f"text_encoder.encoder.layer.{i}."):
                    to_exclude.append(name)
        to_exclude = list(set(to_exclude))
        assert isinstance(self.args, FinetuneArgs)
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_r*2,
            target_modules=lora_modules,
            exclude_modules=to_exclude,
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)

        lora_final_modules = [name for name, module in self.model.named_modules() if hasattr(module, "lora_A")]
        verbose_print(f"🦜 LoRA modules: {lora_final_modules}")
        self.model.print_trainable_parameters()
