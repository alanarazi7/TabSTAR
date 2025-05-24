import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Self

from tabular.constants import BATCH_SIZE
from tabular.datasets.tabular_datasets import OpenMLDatasetID
from tabular.tabstar.params.constants import NumberVerbalization
from tabular.utils.io_handlers import load_json, dump_json
from tabular.utils.logging import LOG_SEP
from tabular.utils.paths import pretrain_args_path, create_dir
from tabular.utils.utils import get_current_commit_hash, get_today, verbose_print

MAX_EPOCH_EXAMPLES = 2048
if MAX_EPOCH_EXAMPLES % BATCH_SIZE != 0:
    raise ValueError(f"MAX_EPOCH_EXAMPLES must be divisible by {BATCH_SIZE}")

# TODO: use HfArgumentParser
@dataclass
class PretrainArgs:
    raw_exp_name: str
    tabular_layers: int
    base_lr: float
    weight_decay: float
    numbers_verbalization: NumberVerbalization
    unfreeze_layers: int
    datasets: List[int]
    fold: Optional[int] = None
    full_exp_name: Optional[str] = None
    cached: bool = False
    num_datasets: int = 0

    def __post_init__(self):
        if self.cached:
            return
        self.num_datasets = len(self.datasets)
        self.full_exp_name = self.set_full_exp_name()

    @classmethod
    def from_args(cls, args: argparse.Namespace, pretrain_data: List[OpenMLDatasetID]) -> Self:
        return PretrainArgs(raw_exp_name=args.exp,
                            tabular_layers=args.tabular_layers,
                            base_lr=args.base_lr,
                            unfreeze_layers=args.e5_unfreeze_layers,
                            weight_decay=args.weight_decay,
                            numbers_verbalization=NumberVerbalization(args.numbers_verbalization),
                            datasets=[d.value for d in pretrain_data],
                            fold=args.fold)

    @classmethod
    def from_json(cls, pretrain_exp: str):
        path = pretrain_args_path(pretrain_exp)
        data = load_json(path)
        verbose_print(f"Loaded the following pretrain args: {data}")
        data['cached'] = True
        args = PretrainArgs(**data)
        args.numbers_verbalization = NumberVerbalization(args.numbers_verbalization)
        if len(args.datasets) == 0:
            assert args.num_datasets == 0, "num_datasets should be 0 if datasets is empty"
        return args

    def to_json(self):
        create_dir(self.path, is_file=True)
        d = asdict(self)
        dump_json(d, self.path)

    @property
    def path(self) -> str:
        return pretrain_args_path(self.full_exp_name)

    def set_full_exp_name(self) -> str:
        strings = [get_today(),
                   self.raw_exp_name,
                   f"data_{self.num_datasets}",
                   f"tab_{self.tabular_layers}",
                   f"layers_{self.unfreeze_layers}",
                   f"num_verb_{self.numbers_verbalization}",
                   f"lr_{str_float(self.base_lr)}",
                   f"wd_{str_float(self.weight_decay)}",
                   f"git_{get_current_commit_hash()}"]
        if self.fold is not None:
            strings.append(f"fold_{self.fold}")
        return LOG_SEP.join(strings)


def str_float(f: float) -> str:
    return str(f).replace('.', '')