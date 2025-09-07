from argparse import Namespace
from dataclasses import dataclass, fields
from typing import Self, Dict


TABULAR_LAYERS = 6
TEXTUAL_UNFREEZE_LAYERS = 6
PRETRAIN_VAL_RATIO = 0.05


# Training
LR = 0.00005
WARMUP = 0.1
WEIGHT_DECAY = 0.001
EPOCHS = 50
EPOCH_EXAMPLES = 2048
PATIENCE = 3
BATCH_SIZE = 32
GLOBAL_BATCH_SIZE = 128


@dataclass
class TrainingArgs:
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup: float = WARMUP
    epochs: int = EPOCHS
    epoch_examples: int = EPOCH_EXAMPLES
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    global_batch_size: int = GLOBAL_BATCH_SIZE
    accumulation_steps: int = 0

    def __post_init__(self):
        if self.batch_size > self.global_batch_size:
            print("Warning: Batch size is greater than global batch size. Setting global batch size to batch size.")
            self.global_batch_size = self.batch_size
        if self.global_batch_size % self.batch_size != 0:
            raise ValueError("Global batch size must be divisible by batch size.")
        self.accumulation_steps = self.global_batch_size // self.batch_size

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