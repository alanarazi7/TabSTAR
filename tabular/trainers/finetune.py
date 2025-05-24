from typing import Type, Optional

import torch
import wandb

from tabular.benchmarks.all_datasets import TEXTUAL_DATASETS
from tabular.datasets.tabular_datasets import TabularDatasetID
from tabular.evaluation.constants import DOWNSTREAM_EXAMPLES
from tabular.models.abstract_model import TabularModel
from tabular.trainers.downstream_train import ModelTrainer, RunMetadata
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.utils.gpus import get_device
from tabular.utils.logging import wandb_run, RunType
from tabular.utils.utils import cprint

def do_finetune_run(exp_name: str,
                    model: Type[TabularModel],
                    dataset: TabularDatasetID,
                    run_num: int,
                    train_examples: int = DOWNSTREAM_EXAMPLES,
                    device: Optional[torch.device] = None,
                    finetune_args: Optional[FinetuneArgs] = None,
                    carte_lr_idx: Optional[int] = None) -> RunMetadata:
    if device is None:
        device = torch.device(get_device())
    if isinstance(finetune_args, FinetuneArgs):
        if finetune_args.pretrain_args.datasets:
            if dataset.value in finetune_args.pretrain_args.datasets:
                raise RuntimeError(f"Can't finetune model with dataset {dataset} that appears in pretraining!")
    trainer = ModelTrainer(dataset_id=dataset, model_cls=model, exp_name=exp_name, device=device,
                           run_num=run_num, args=finetune_args, train_examples=train_examples,
                           carte_lr_idx=carte_lr_idx)
    if run_metadata := trainer.existing_score():
        cprint(f"Already trained {model.MODEL_NAME} on {dataset.name} for run {run_num}: {run_metadata.test_score:.3f}")
        return run_metadata
    run_type = RunType.FINETUNE if finetune_args else RunType.BASELINE
    wandb_run(trainer.run_name, run_type=run_type)
    cprint(f"🏆 Training {dataset} on baseline: {trainer.run_name}")
    run_metadata = trainer.run()
    cprint(f"Run: {trainer.run_name}\n💯 Score: {run_metadata.test_score:.3f}")
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
