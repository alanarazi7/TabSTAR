import os
import shutil
import time
from os.path import join, exists
from random import sample
from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from tabstarv1p1_pretrain.finetunability.args import FinetunabilityArgs
from tabstarv1p1_pretrain.finetunability.finetunability_check import assert_finetunability_installed, run_finetunability_check
from tabstar.tabstarv1p1.finetune.lora_params import assert_lora_coverage
from tabstarv1p1_pretrain.training.dataloaders import MixedBatchesSampler
from tabstarv1p1_pretrain.training.trainer import update_model
from tabstar.tabstarv1p1.training.optimizer import get_epoch_scheduler
from tabstar.tabstarv1p1.training.utils import fix_seed
from tabstar.tabstarv1p1.arch.model import TabSTARModel
from tabstar.tabstarv1p1.arch.config import TabStarConfig
from tabstarv1p1_pretrain.pretraining.datasets import get_pretrain_multi_dataloader
from tabstarv1p1_pretrain.pretraining.hdf5 import RealDataset
from tabstarv1p1_pretrain.pretraining.logging import log_epoch_metrics, print_epoch_summary, log_finetunability, \
    log_training_complete
from tabstarv1p1_pretrain.pretraining.pretrain_args import PretrainArgs, TaskType, PoolMode
from tabstarv1p1_pretrain.pretraining.paths import get_pretrain_checkpoint, \
    get_finetunability_snapshot_path, pretrain_exp_dir
from tabstar.tabstarv1p1.training.loss import calculate_mixed_loss
from tabstar.tabstarv1p1.training.batching import TabSTARBatch
from tabstarv1p1_pretrain.utils.devices import assert_bf16_supported
from tabstar.tabstarv1p1.training.devices import clear_cuda_cache
from tabstarv1p1_pretrain.constants import MEMORY
from tabstarv1p1_pretrain.pretraining.hyperparameters import TrainingArgs
from tabstarv1p1_pretrain.pretraining.logging import summarize_model, log_epoch_start
from tabstarv1p1_pretrain.pretraining.monitoring import print_gpu_mem_gb, reset_peak_gpu_mem, peak_gpu_mem_gb
from tabstarv1p1_pretrain.pretraining.optimizer import get_optimizer
from tabstarv1p1_pretrain.pretraining.unfreezing import unfreeze_text_encoder
from tabstarv1p1_pretrain.utils.timing import elapsed_minutes


def list_t4_task_dirs(t4_dir: str, task: str) -> List[str]:
    task_dir = join(t4_dir, task)
    if not exists(task_dir):
        raise FileNotFoundError(f"No '{task}' subdirectory under T4 dir {t4_dir}")
    return sorted(
        join(task_dir, name) for name in os.listdir(task_dir)
        if exists(join(task_dir, name, RealDataset.PROPERTIES))
    )


def sample_t4_dirs(t4_dir: str, quotas: List[Tuple[str, int]],
                   pool_mode: PoolMode) -> Tuple[List[str], Dict[str, int]]:
    """Returns the sampled dirs and how many datasets each task's pool holds."""
    t4_dirs = []
    pool_sizes = {}
    for task, quota in quotas:
        task_dirs = list_t4_task_dirs(t4_dir, task=task)
        pool_sizes[task] = len(task_dirs)
        if len(task_dirs) < quota:
            raise ValueError(f"Requested {quota} T4 '{task}' datasets from {t4_dir}, "
                             f"only {len(task_dirs)} available")
        if pool_mode == PoolMode.FIRST:
            t4_dirs.extend(task_dirs[:quota])
            print(f"🥇 Took first {quota} of {len(task_dirs)} T4 '{task}' datasets from {t4_dir}")
        else:
            t4_dirs.extend(sample(task_dirs, k=quota))
            print(f"🎲 Sampled {quota} of {len(task_dirs)} T4 '{task}' datasets from {t4_dir}")
    return t4_dirs, pool_sizes


class TabSTARPretrainer:

    def __init__(self,
                 run_name: str,
                 device: torch.device,
                 train_args: TrainingArgs,
                 pretrain_args: PretrainArgs,
                 finetunability_args: Optional[FinetunabilityArgs] = None):
        self.train_args = train_args
        self.run_name = run_name
        self.device = device
        self.args = pretrain_args
        self.finetunability_args = finetunability_args or FinetunabilityArgs()
        assert_finetunability_installed()
        self.data_dirs: List[str] = []
        self.num_t4_datasets_used: int = 0
        self.t4_quotas: List[Tuple[str, int]] = []
        self.t4_pool_sizes: Dict[str, int] = {}
        self.best_finetunability: Dict[str, float] = {}
        assert_bf16_supported(device=self.device)
        fix_seed()
        self.model: TabSTARModel = self.initialize_model()
        assert_lora_coverage(self.model)
        self.optimizer: AdamW = get_optimizer(model=self.model, args=self.train_args)
        # TODO: total_steps=epochs, so the one-cycle schedule ignores accumulation steps. It also hit
        # a torch 2.11 ZeroDivisionError at exactly total_steps=10 (pct_start=0.1, so
        # pct_start * total_steps == 1.0); the tabstar helper retries with epochs + 1 instead of
        # using a real step count, which makes such exact-integer boundaries easy to stumble into.
        self.scheduler: LRScheduler = get_epoch_scheduler(optimizer=self.optimizer,
                                                          max_lr=self.train_args.learning_rate,
                                                          epochs=self.train_args.epochs)
        self.add_t4_data_dirs()
        self.steps: int = 0
        self.epoch: int = 0

    def initialize_model(self) -> TabSTARModel:
        config = TabStarConfig(reg_dist_bins=self.args.reg_dist_bins)
        model = TabSTARModel(config=config)
        model.text_batch_size = self.args.text_batch_size
        unfreeze_text_encoder(text_encoder=model.text_encoder, layers_to_unfreeze=self.args.unfreeze_layers)
        model.to(self.device)
        return model

    def add_t4_data_dirs(self):
        if not self.args.t4_dir:
            return
        if self.args.task_type == TaskType.CLS:
            self.t4_quotas = [("cls", self.args.num_datasets)]
        else:
            self.t4_quotas = [("reg", self.args.num_datasets)]
        t4_dirs, self.t4_pool_sizes = sample_t4_dirs(self.args.t4_dir, quotas=self.t4_quotas,
                                                     pool_mode=self.args.pool_mode)
        self.data_dirs.extend(t4_dirs)
        self.num_t4_datasets_used = len(t4_dirs)
        print(f"🌟 Added {len(t4_dirs)} T4 datasets ({self.args.task_type}, {self.args.pool_mode}) "
              f"from {self.args.t4_dir}")

    def refresh_t4_data_dirs(self):
        t4_dirs, self.t4_pool_sizes = sample_t4_dirs(self.args.t4_dir, quotas=self.t4_quotas,
                                                     pool_mode=self.args.pool_mode)
        self.data_dirs = t4_dirs
        self.num_t4_datasets_used = len(t4_dirs)

    def train(self):
        t0 = time.time()
        summarize_model(self.model)
        print(f"💪 Pretraining for {self.run_name} over {len(self.data_dirs)} datasets on device {self.device}.")
        if self.epoch == 0 and not self.finetunability_args.skip_epoch_zero:
            # A fresh start (not a resume): get the finetunability signal's own starting point,
            # before any pretrain epoch has touched the model.
            self.track_finetunability(epoch=0)
        dataloader = get_pretrain_multi_dataloader(data_dirs=self.data_dirs, train_args=self.train_args)
        assert isinstance(dataloader.batch_sampler, MixedBatchesSampler)
        with tqdm(total=self.train_args.epochs, desc="Epochs", leave=False) as pbar_epochs:
            for epoch in range(self.epoch + 1, self.train_args.epochs + 1):
                self.epoch = epoch
                if self.args.pool_mode == PoolMode.ACTIVE and epoch > 1:
                    self.refresh_t4_data_dirs()
                    dataloader = get_pretrain_multi_dataloader(data_dirs=self.data_dirs, train_args=self.train_args)
                    assert isinstance(dataloader.batch_sampler, MixedBatchesSampler)
                log_epoch_start(scheduler=self.scheduler, steps=self.steps, epoch=self.epoch)
                dataloader.batch_sampler.make_batches()
                train_loss = torch.zeros((), device=self.device)  # 0-dim scalar, stays on-device until epoch end
                train_examples = 0
                reset_peak_gpu_mem(self.device)
                train_t0 = time.time()
                with tqdm(total=len(dataloader), desc="Batches", leave=False) as pbar_batches:
                    for batch_idx, batch in enumerate(dataloader):
                        assert isinstance(batch, TabSTARBatch)
                        batch_loss = self.train_one_batch(batch=batch)
                        train_loss += batch_loss * len(batch.y)
                        train_examples += len(batch.y)
                        self.steps += 1
                        if (batch_idx + 1) % self.train_args.accumulation_steps == 0:
                            self.do_update()
                        pbar_batches.update(1)
                    if (batch_idx + 1) % self.train_args.accumulation_steps != 0:
                        self.do_update()
                train_minutes = elapsed_minutes(train_t0)
                # Single CPU-GPU sync for the whole epoch, rather than one per micro-batch.
                train_loss = (train_loss / train_examples).item()
                elapsed = time.time() - t0
                peak_mem = peak_gpu_mem_gb(self.device)
                log_epoch_metrics(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                                  train_minutes=train_minutes, device=self.device)
                print_epoch_summary(epoch=epoch, train_minutes=train_minutes, total_minutes=elapsed / 60,
                                    train_loss=train_loss, peak_mem_gb=peak_mem)
                self.track_finetunability(epoch=epoch)
                self.scheduler.step()
                pbar_epochs.update(1)
                self.save_checkpoint()
        log_training_complete(epoch=epoch)

    def do_forward(self, batch: TabSTARBatch) -> Tuple[Tensor, Tensor]:
        batch.to_device(self.device, dtype=torch.bfloat16)
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            predictions = self.model(batch)
        loss = calculate_mixed_loss(y=batch.y, predictions=predictions, d_outputs=batch.d_outputs,
                                    bin_edges=self.model.reg_head.bin_edges,
                                    bin_centers=self.model.reg_head.bin_centers)
        return predictions, loss

    def train_one_batch(self, batch: TabSTARBatch) -> Tensor:
        self.model.train()
        return self.train_batch_with_oom_retry(batch, weight=1.0)

    def train_batch_with_oom_retry(self, batch: TabSTARBatch, weight: float) -> Tensor:
        """Forward and backward OOM retries by splitting the batch in half and recursing, each
        half's loss weighted by its share of the original batch so the accumulated gradient
        matches what a single, un-split backward pass over the full batch would have produced.
        A backward call that OOMs partway through may already have applied gradients for part of
        the graph, so on that failure the current accumulation window's gradients are wiped via
        zero_grad() before retrying, rather than assuming which parameters got a partial update.
        """
        try:
            predictions, loss = self.do_forward(batch)
        except torch.cuda.OutOfMemoryError:
            return self.retry_split(batch, weight)
        if not torch.isfinite(loss):
            print(f"⚠️ Non-finite loss ({loss.item()}) at step {self.steps}; skipping batch without backward.")
            return torch.zeros((), device=loss.device)
        loss_for_backward = loss * weight / self.train_args.accumulation_steps
        try:
            loss_for_backward.backward()
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad()
            # The failed backward's forward-pass graph (predictions, loss, loss_for_backward)
            # would otherwise stay referenced by this frame for the entire retry recursion below.
            del predictions, loss, loss_for_backward
            clear_cuda_cache()
            return self.retry_split(batch, weight)
        return loss.detach()

    def retry_split(self, batch: TabSTARBatch, weight: float) -> Tensor:
        batch_size = len(batch.y)
        if batch_size == 1:
            raise torch.cuda.OutOfMemoryError
        clear_cuda_cache()
        mid = batch_size // 2
        remainder = batch_size - mid
        print(f"🤯 Splitting training batch of {batch_size} in half due to OOM")
        first_weight = weight * mid / batch_size
        second_weight = weight * remainder / batch_size
        first_loss = self.train_batch_with_oom_retry(batch.slice(0, mid), weight=first_weight)
        second_loss = self.train_batch_with_oom_retry(batch.slice(mid, batch_size), weight=second_weight)
        return (first_loss * mid + second_loss * remainder) / batch_size

    def do_update(self):
        update_model(model=self.model, optimizer=self.optimizer)
        if MEMORY and self.steps % 50 == 0:
            print(f"Step {self.steps} Mem {print_gpu_mem_gb(self.device)}")

    def save_checkpoint(self):
        save_path = get_pretrain_checkpoint(self.run_name, epoch=self.epoch)
        self.model.save_pretrained(save_path)

    def track_finetunability(self, epoch: int):
        args = self.finetunability_args
        if epoch % args.every_n_epochs != 0:
            return
        finetunability_t0 = time.time()
        snapshot_path = get_finetunability_snapshot_path(self.run_name, epoch=epoch)
        self.model.save_pretrained(snapshot_path)
        raw_scores_dir = join(pretrain_exp_dir(self.run_name), "finetunability_raw")
        try:
            scores = run_finetunability_check(checkpoint_dir=snapshot_path,
                                              run_name=f"{self.run_name}_finetunability_ep{epoch}",
                                              timeout_s=args.timeout_s,
                                              raw_scores_dir=raw_scores_dir,
                                              task_type=self.args.task_type)
        finally:
            shutil.rmtree(snapshot_path, ignore_errors=True)
        minutes = elapsed_minutes(finetunability_t0)
        if scores is None:
            raise RuntimeError(f"Epoch {epoch}: finetunability check failed after {minutes:.1f}m; "
                                f"see the warning above and {raw_scores_dir} for details.")
        for key in ("elo", "cls_elo", "reg_elo"):
            value = scores.get(key)
            if value is not None:
                self.best_finetunability[key] = max(value, self.best_finetunability.get(key, value))
        log_finetunability(epoch=epoch, scores=scores, minutes=minutes, best_scores=self.best_finetunability)