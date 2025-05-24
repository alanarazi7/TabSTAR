import argparse
from dataclasses import dataclass
from os.path import join
from typing import Optional, Self

from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.logging import LOG_SEP


# TODO: use HfArgumentParser
@dataclass
class FinetuneArgs:
    raw_exp_name: str
    pretrain_args: PretrainArgs
    lora_lr: float
    lora_batch: int
    lora_r: int
    patience: int
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
                            keep_model=args.downstream_keep_model,
                            patience=args.downstream_patience)

    def set_full_exp_name(self) -> str:
        strings = [self.raw_exp_name,
                   f"lora_lr_{self.lora_lr}",
                   f"lora_batch_{self.lora_batch}",
                   f"lora_r_{self.lora_r}",
                   f"patience_{self.patience}"]
        finetune_exp = LOG_SEP.join(strings)
        return join(self.pretrain_args.full_exp_name, finetune_exp)
