import argparse
from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import Dict, Optional, Self

from tabstar.tabstarv1p1.constants import TEXT_BATCH_SIZE
from tabstarv1p1_pretrain.finetunability.args import FinetunabilityArgs
from tabstarv1p1_pretrain.pretraining.arguments import REG_DIST_BINS
from tabstarv1p1_pretrain.pretraining.paths import pretrain_args_path
from tabstarv1p1_pretrain.pretraining.hyperparameters import TrainingArgs
from tabstarv1p1_pretrain.utils.io_handlers import dump_json
from tabstarv1p1_pretrain.utils.timing import get_now


MIN_REG_DATASETS = 10_000


class TaskType(StrEnum):
    CLS = "cls"
    REG = "reg"


class PoolMode(StrEnum):
    RANDOM = "random"
    ACTIVE = "active"
    FIRST = "first"

@dataclass
class PretrainArgs:
    raw_exp_name: str
    unfreeze_layers: int
    timestamp: str
    num_datasets: int
    task_type: TaskType
    text_batch_size: int = TEXT_BATCH_SIZE
    t4_dir: Optional[str] = None
    pool_mode: PoolMode = PoolMode.ACTIVE
    t4_pool_sizes: Optional[Dict[str, int]] = None
    reg_dist_bins: int = REG_DIST_BINS

    def __post_init__(self):
        if self.task_type not in (TaskType.CLS, TaskType.REG):
            raise ValueError(f"task_type must be one of {list(TaskType)}, got {self.task_type!r}")
        if self.task_type == TaskType.REG and self.num_datasets < MIN_REG_DATASETS:
            raise ValueError(f"Regression pretraining requires --num_datasets >= {MIN_REG_DATASETS}, "
                             f"got {self.num_datasets}")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        return PretrainArgs(raw_exp_name=args.exp,
                            unfreeze_layers=args.e5_unfreeze_layers,
                            timestamp=get_now(),
                            text_batch_size=args.text_batch_size,
                            num_datasets=args.num_datasets,
                            t4_dir=args.t4_dir,
                            task_type=TaskType(args.task_type),
                            pool_mode=PoolMode(args.pool_mode),
                            reg_dist_bins=args.reg_dist_bins)

    @property
    def full_exp_name(self) -> str:
        return f"{self.timestamp}_{self.raw_exp_name}"

    @property
    def path(self) -> str:
        return pretrain_args_path(self.full_exp_name)


@dataclass
class PretrainConfig:
    """Full set of arguments behind a pretrain run, dumped for provenance. Nothing reads it back."""
    pretrain_args: PretrainArgs
    train_args: TrainingArgs
    finetunability_args: FinetunabilityArgs

    @property
    def path(self) -> str:
        return self.pretrain_args.path

    def to_json(self) -> None:
        dump_json(asdict(self), self.path)
