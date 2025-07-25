from typing import Type, Optional

import torch
import wandb

from tabstar_paper.benchmarks.text_benchmarks import TEXTUAL_DATASETS
from tabstar_paper.utils.logging import wandb_run
from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.models.abstract_model import TabularModel
from tabular.trainers.downstream_train import ModelTrainer, RunMetadata
from tabular.trainers.finetune_args import FinetuneArgs


def do_finetune_run(exp_name: str,
                    model: Type[TabularModel],
                    dataset: TabularDatasetID,
                    run_num: int,
                    train_examples: int,
                    device: torch.device,
                    finetune_args: Optional[FinetuneArgs] = None,
                    carte_lr_idx: Optional[int] = None) -> RunMetadata:
    trainer = ModelTrainer(dataset_id=dataset, model_cls=model, exp_name=exp_name, device=device,
                           run_num=run_num, train_examples=train_examples, carte_lr_idx=carte_lr_idx)
    if run_metadata := trainer.existing_score():
        print(f"Already trained {model.MODEL_NAME} on {dataset.name} for run {run_num}: {run_metadata.test_score:.3f}")
        return run_metadata
    wandb_run(trainer.run_name, project="tabstar_baseline")
    print(f"🏆 Training {dataset} on baseline: {trainer.run_name}")
    run_metadata = trainer.run()
    print(f"Run: {trainer.run_name}\n💯 Score: {run_metadata.test_score:.3f}")
    results = {'model': model.MODEL_NAME,
               'dataset': dataset.name,
               'score': run_metadata.test_score,
               'dev_loss': run_metadata.dev_loss,
               'run_num': run_num,
               'dataset_size': train_examples,
               'is_text': bool(dataset in TEXTUAL_DATASETS),
               'carte_lr_idx': carte_lr_idx,
               'pretrain_model': finetune_args.pretrain_args.full_exp_name if finetune_args else None,
               'finetune_exp': exp_name,
               'finetune_raw_exp': finetune_args.raw_exp_name if finetune_args else None, }
    wandb.log(results)
    wandb.summary.update(results)
    wandb.finish()
    return run_metadata
