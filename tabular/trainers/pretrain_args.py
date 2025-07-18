import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Self, Any

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar_paper.pretraining.paths import pretrain_args_path
from tabstar_paper.utils.io_handlers import load_json, dump_json
from tabstar_paper.utils.timing import get_now

from tabular.utils.paths import create_dir
from tabular.utils.utils import verbose_print


# TODO: use HfArgumentParser
@dataclass
class PretrainArgs:
    raw_exp_name: str
    tabular_layers: int
    base_lr: float
    weight_decay: float
    unfreeze_layers: int
    datasets: List[Any]
    epochs: int
    timestamp: str
    num_datasets: int
    fold: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace, pretrain_data: List[TabularDatasetID]) -> Self:
        num_datasets = len(pretrain_data)
        return PretrainArgs(raw_exp_name=args.exp,
                            tabular_layers=args.tabular_layers,
                            base_lr=args.base_lr,
                            unfreeze_layers=args.e5_unfreeze_layers,
                            weight_decay=args.weight_decay,
                            datasets=[d.value for d in pretrain_data],
                            epochs=args.epochs,
                            fold=args.fold,
                            timestamp=get_now(),
                            num_datasets=num_datasets)

    @classmethod
    def from_json(cls, pretrain_exp: str):
        path = pretrain_args_path(pretrain_exp)
        data = load_json(path)
        verbose_print(f"Loaded the following pretrain args: {data}")
        args = PretrainArgs(**data)
        if len(args.datasets) == 0:
            assert args.num_datasets == 0, "num_datasets should be 0 if datasets is empty"
        return args

    def to_json(self):
        create_dir(self.path, is_file=True)
        d = asdict(self)
        dump_json(d, self.path)

    @property
    def full_exp_name(self) -> str:
        return f"{self.timestamp}_{self.raw_exp_name}"

    @property
    def path(self) -> str:
        return pretrain_args_path(self.full_exp_name)
