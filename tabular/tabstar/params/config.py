from typing import Self

from transformers import PretrainedConfig

from tabular.constants import BATCH_SIZE
from tabular.tabstar.params.constants import (TABULAR_LAYERS, GLOBAL_BATCH_SIZE,
                                              TEXTUAL_UNFREEZE_LAYERS, BASE_LR, WEIGHT_DECAY, D_MODEL)
from tabular.trainers.finetune_args import FinetuneArgs
from tabular.trainers.pretrain_args import PretrainArgs
from tabular.utils.utils import cprint, verbose_print


class TabStarConfig(PretrainedConfig):
    model_type = "tabstar"

    def __init__(
        self,
        is_pretrain: bool = True,
        num_layers: int = TABULAR_LAYERS,
        unfreeze_layers: int = TEXTUAL_UNFREEZE_LAYERS,
        lr: float = BASE_LR,
        weight_decay: float = WEIGHT_DECAY,
        macro_batch_size: int = GLOBAL_BATCH_SIZE,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = D_MODEL
        self.is_pretrain = is_pretrain
        self.unfreeze_layers = unfreeze_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.macro_batch_size = macro_batch_size
        self.batch_size = batch_size

    @property
    def accumulation_steps(self) -> int:
        accumulation_steps = self.macro_batch_size // self.batch_size
        assert accumulation_steps * self.batch_size == self.macro_batch_size
        verbose_print(f"👣 Using accumulation steps of {accumulation_steps}")
        return accumulation_steps


    @classmethod
    def create(cls, args: FinetuneArgs | PretrainArgs) -> Self:
        if isinstance(args, PretrainArgs):
            is_pretrain = True
            pretrain_args = args
        elif isinstance(args, FinetuneArgs):
            is_pretrain = False
            pretrain_args = args.pretrain_args
        else:
            raise TypeError(f"Expected FinetuneArgs or PretrainArgs, got {type(args)}")
        config = cls(
            is_pretrain=is_pretrain,
            num_layers=pretrain_args.tabular_layers,
            d_model=D_MODEL,
            unfreeze_layers=pretrain_args.unfreeze_layers,
            lr=pretrain_args.base_lr,
            weight_decay=pretrain_args.weight_decay,
        )
        if not is_pretrain:
            config.adjust_for_finetune(args)
        config.print_lrs()
        return config

    def print_lrs(self):
        cprint(f"🤓 Using LR of {self.lr=}, with {self.batch_size=} and {self.macro_batch_size=}!")

    def adjust_for_finetune(self, args: FinetuneArgs):
        self.lr = args.lora_lr
        self.batch_size = args.lora_batch
        self.macro_batch_size = max(args.lora_batch, GLOBAL_BATCH_SIZE)
