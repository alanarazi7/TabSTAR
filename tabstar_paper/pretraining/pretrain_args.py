import argparse
from dataclasses import dataclass, asdict, fields
from typing import List, Optional, Self, Any

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar_paper.pretraining.paths import pretrain_args_path
from tabstar_paper.utils.io_handlers import load_json, dump_json, create_dir
from tabstar_paper.utils.timing import get_now


# TODO: slowly deprecate this?
@dataclass
class PretrainArgs:
    raw_exp_name: str
    tabular_layers: int
    unfreeze_layers: int
    datasets: List[Any]
    timestamp: str
    num_datasets: int
    fold: Optional[int] = None
    checkpoint: Optional[int] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace, pretrain_data: List[TabularDatasetID]) -> Self:
        num_datasets = len(pretrain_data)
        return PretrainArgs(raw_exp_name=args.exp,
                            tabular_layers=args.tabular_layers,
                            unfreeze_layers=args.e5_unfreeze_layers,
                            datasets=[d.value for d in pretrain_data],
                            fold=args.fold,
                            timestamp=get_now(),
                            num_datasets=num_datasets,
                            checkpoint=args.checkpoint)

    @classmethod
    def from_json(cls, pretrain_exp: str):
        path = pretrain_args_path(pretrain_exp)
        data = load_json(path)
        allowed_fields = {f.name for f in fields(PretrainArgs)}
        data = {k: v for k, v in data.items() if k in allowed_fields}
        args = PretrainArgs(**data)
        if len(args.datasets) == 0:
            assert args.num_datasets == 0, "num_datasets should be 0 if datasets is empty"
        return args

    def to_json(self):
        create_dir(self.path)
        d = asdict(self)
        dump_json(d, self.path)

    @property
    def full_exp_name(self) -> str:
        if self.checkpoint:
            return self.raw_exp_name
        return f"{self.timestamp}_{self.raw_exp_name}"

    @property
    def path(self) -> str:
        return pretrain_args_path(self.full_exp_name)
