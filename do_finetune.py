import argparse
from dataclasses import dataclass
from os.path import join
from typing import Optional

import torch
import wandb

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARRegressor, TabSTARClassifier
from tabstar.training.devices import get_device
from tabstar.training.hyperparams import LORA_LR, LORA_R, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_BATCH
from tabstar_paper.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, FOLDS
from tabstar_paper.constants import DEVICE
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg
from tabstar_paper.preprocessing.sampling import subsample_dataset
from tabstar_paper.pretraining.paths import get_model_path
from tabstar_paper.pretraining.pretrain_args import PretrainArgs
from tabstar_paper.utils.logging import wandb_run

# TODO: unsure this class is needed at all
@dataclass
class FinetuneArgs:
    raw_exp_name: str
    pretrain_args: PretrainArgs
    lora_lr: float
    lora_batch: int
    lora_r: int
    patience: int
    epochs: int
    keep_model: bool = False
    full_exp_name: Optional[str] = None

    def __post_init__(self):
        self.full_exp_name = self.set_full_exp_name()

    @classmethod
    def from_args(cls, args: argparse.Namespace, exp_name: str, pretrain_args: PretrainArgs) -> Self:
        return FinetuneArgs(raw_exp_name=exp_name,
                            pretrain_args=pretrain_args,
                            lora_lr=args.lora_lr,
                            lora_batch=args.lora_batch,
                            lora_r=args.lora_r,
                            epochs=args.epochs,
                            keep_model=args.downstream_keep_model,
                            patience=args.patience)

    def set_full_exp_name(self) -> str:
        strings = [self.raw_exp_name,
                   f"epochs_{self.epochs}",
                   f"lora_lr_{self.lora_lr}",
                   f"lora_batch_{self.lora_batch}",
                   f"lora_r_{self.lora_r}",
                   f"patience_{self.patience}"]
        finetune_exp = "__".join(strings)
        return join(self.pretrain_args.full_exp_name, finetune_exp)


def finetune_tabstar(finetune_args: FinetuneArgs,
                     dataset_id: TabularDatasetID,
                     fold: int,
                     train_examples: int,
                     device: torch.device,
                     verbose: bool = False):
    wandb_run(exp_name=finetune_args.full_exp_name, project="tabstar_finetune")
    if dataset_id.value in finetune_args.pretrain_args.datasets:
        raise RuntimeError(f"😱 Dataset {dataset_id} is already in pretrain datasets, beware!")
    dataset = download_dataset(dataset_id=dataset_id)
    is_cls = dataset.is_cls
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=is_cls, train_examples=train_examples, fold=fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=fold, train_examples=train_examples)
    if is_cls:
        tabstar_cls = TabSTARClassifier
    else:
        tabstar_cls = TabSTARRegressor
    pretrain_exp = finetune_args.pretrain_args.full_exp_name
    pretrain_path = get_model_path(pretrain_exp)
    model = tabstar_cls(pretrain_dataset_or_path=pretrain_path,
                        lora_lr=finetune_args.lora_lr,
                        lora_r=finetune_args.lora_r,
                        max_epochs=finetune_args.epochs,
                        lora_batch=finetune_args.lora_batch,
                        patience=finetune_args.patience,
                        verbose=verbose,
                        device=device)
    model.fit(x_train, y_train)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    print(f"Scored {metrics.score:.4f} on dataset {dataset_id}.")
    results = {"model": "TabSTAR 🌟",
               "dataset_id": dataset_id.name,
               "dataset_value": dataset_id.value,
               "test_score": metrics.score,
               "fold": fold,
               "train_examples": train_examples,
               "pretrain_model": finetune_args.pretrain_args.full_exp_name,
               "finetune_exp": finetune_args.raw_exp_name}
    wandb.log(results)
    wandb.summary.update(results)
    wandb.finish()
    return metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_exp', type=str, required=True)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--exp', type=str, default="default_finetune_exp")
    parser.add_argument('--downstream_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--downstream_keep_model', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    # Training
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=FINETUNE_PATIENCE)
    parser.add_argument('--lora_lr', type=float, default=LORA_LR)
    parser.add_argument('--lora_batch', type=int, default=LORA_BATCH)
    parser.add_argument('--lora_r', type=int, default=LORA_R)

    args = parser.parse_args()
    assert args.pretrain_exp, "Pretrain path is required"
    data = get_dataset_from_arg(args.dataset_id)
    assert 0 <= args.fold < FOLDS, f"Invalid run number: {args.fold}. Should be between 0 and {FOLDS - 1}"

    pretrain_args = PretrainArgs.from_json(pretrain_exp=args.pretrain_exp)
    run_args = FinetuneArgs.from_args(args=args, pretrain_args=pretrain_args, exp_name=args.exp)
    my_device = get_device(device=DEVICE)
    finetune_tabstar(finetune_args=run_args, dataset_id=data, fold=args.fold, train_examples=args.downstream_examples,
                     device=my_device, verbose=args.verbose)
