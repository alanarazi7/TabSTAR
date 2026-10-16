from argparse import Namespace
from dataclasses import fields
from typing import Dict, Any

from tabstar.tabstarv1p1.constants import INTERACTION_LAYERS, D_MODEL, D_NUM, TEXT_MODEL, REG_Z_CLIP, REG_DIST_BINS

# Data
DATASET_TRAIN_SIZE = 1024
DATASET_VAL_SIZE = 64


def init_dataclass(args: Namespace | Dict, cls: Any, name: str) -> Any:
    if isinstance(args, Namespace):
        args = vars(args)
    valid_keys = {f.name for f in fields(cls)}
    filtered_args = {k: v for k, v in args.items() if k in valid_keys}
    print(f"{name} args:")
    for k, v in filtered_args.items():
        print(f"  - {k}: {v}")
    return cls(**filtered_args)
