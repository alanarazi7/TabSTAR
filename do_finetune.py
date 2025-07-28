import argparse

import torch

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.tabstar_model import TabSTARRegressor, TabSTARClassifier
from tabstar.training.devices import get_device
from tabstar.training.hyperparams import LORA_LR, LORA_R, MAX_EPOCHS, FINETUNE_PATIENCE, LORA_BATCH
from tabstar_paper.benchmarks.evaluate import DOWNSTREAM_EXAMPLES, TRIALS
from tabstar_paper.constants import DEVICE
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg
from tabstar_paper.preprocessing.sampling import subsample_dataset

# TODO: remove tabular imports
from tabular.constants import VERBOSE
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.paths import get_model_path


def finetune_tabstar(finetune_args: FinetuneArgs,
                     dataset_id: TabularDatasetID,
                     trial: int,
                     train_examples: int,
                     device: torch.device):
    if dataset_id.value in finetune_args.pretrain_args.datasets:
        raise RuntimeError(f"😱 Dataset {dataset_id} is already in pretrain datasets, beware!")
    dataset_id = download_dataset(dataset_id=dataset_id)
    is_cls = dataset_id.is_cls
    x, y = subsample_dataset(x=dataset_id.x, y=dataset_id.y, is_cls=is_cls, train_examples=train_examples, trial=trial)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, trial=trial, train_examples=train_examples)
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
                        patience=finetune_args.patience,
                        verbose=VERBOSE,
                        device=device)
    model.fit(x_train, y_train)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    print(f"Scored {metrics.score:.4f} on dataset {dataset_id.dataset_id}.")
    return metrics




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_exp', type=str, required=True)
    parser.add_argument('--dataset_id', required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--exp', type=str, default="default_finetune_exp")
    parser.add_argument('--downstream_examples', type=int, default=DOWNSTREAM_EXAMPLES)
    parser.add_argument('--downstream_keep_model', action='store_true', default=False)
    # Training
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=FINETUNE_PATIENCE)
    parser.add_argument('--lora_lr', type=float, default=LORA_LR)
    parser.add_argument('--lora_batch', type=int, default=LORA_BATCH)
    parser.add_argument('--lora_r', type=int, default=LORA_R)

    args = parser.parse_args()
    assert args.pretrain_exp, "Pretrain path is required"
    data = get_dataset_from_arg(args.dataset_id)
    assert 0 <= args.trial < TRIALS, f"Invalid run number: {args.trial}. Should be between 0 and {TRIALS - 1}"

    pretrain_args = PretrainArgs.from_json(pretrain_exp=args.pretrain_exp)
    run_args = FinetuneArgs.from_args(args=args, pretrain_args=pretrain_args, exp_name=args.exp)
    my_device = get_device(device=DEVICE)
    finetune_tabstar(finetune_args=run_args, dataset_id=data,
                     trial=args.trial, train_examples=args.downstream_examples, device=my_device)
