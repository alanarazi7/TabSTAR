from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, Self


@dataclass
class FinetunabilityArgs:
    every_n_epochs: int = 1
    timeout_s: float = 1800.0
    skip_epoch_zero: bool = True

    @classmethod
    def from_args(cls, args: Namespace | Dict) -> Self:
        from tabstarv1p1_pretrain.pretraining.arguments import init_dataclass

        return init_dataclass(args=args, cls=cls, name="🏟️ Finetunability")
