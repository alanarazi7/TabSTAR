from argparse import Namespace
from dataclasses import dataclass, fields
from typing import Self, Dict

from tabstar.tabstarv1p1.constants import TEXTUAL_UNFREEZE_LAYERS

PRETRAIN_VAL_RATIO = 0.05

# T4 datasets never exceed this; guards the assumption behind per-epoch batch construction.
MAX_DATASET_ROWS = 2048


# Training
LR = 0.0001
WARMUP = 0.1
WEIGHT_DECAY = 0.001
EPOCHS = 25
PATIENCE = 5
MICRO_BATCH_SIZE = 32
BATCH_SIZE = 128
GLOBAL_BATCH_SIZE = 512


@dataclass
class TrainingArgs:
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup: float = WARMUP
    epochs: int = EPOCHS
    patience: int = PATIENCE
    micro_batch_size: int = MICRO_BATCH_SIZE
    batch_size: int = BATCH_SIZE
    accumulation_steps: int = 0

    def __post_init__(self):
        if self.batch_size > GLOBAL_BATCH_SIZE:
            raise ValueError(f"batch_size {self.batch_size} exceeds GLOBAL_BATCH_SIZE {GLOBAL_BATCH_SIZE}")
        if GLOBAL_BATCH_SIZE % self.batch_size != 0:
            raise ValueError("GLOBAL_BATCH_SIZE must be divisible by batch_size.")
        self.accumulation_steps = GLOBAL_BATCH_SIZE // self.batch_size

    @classmethod
    def from_args(cls, args: Namespace | Dict) -> Self:
        if isinstance(args, Namespace):
            args = vars(args)
        valid_keys = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        print("🏋️ Pretraining with args:")
        for k, v in filtered_args.items():
            print(f"  - {k}: {v}")
        return cls(**filtered_args)