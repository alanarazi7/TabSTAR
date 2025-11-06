from typing import Optional

from tabstar.datasets.benchmark_folds import TEXT2FOLD
from tabstar.datasets.pretrain_folds import PRETRAIN2FOLD


def get_tabstar_version(pretrain_dataset_or_path: Optional[str] = None) -> str:
    if pretrain_dataset_or_path is None:
        return "alana89/TabSTAR"
    if pretrain_dataset_or_path.startswith(("BIN_", "REG_", "MUL_")):
        tabstar_version = get_tabstar_version_from_dataset(pretrain_dataset=pretrain_dataset_or_path)
        return f"alana89/{tabstar_version}"
    if isinstance(pretrain_dataset_or_path, str):
        return pretrain_dataset_or_path
    raise ValueError(f"Unknown pretrain_dataset_or_path: {pretrain_dataset_or_path}")


def get_tabstar_version_from_dataset(pretrain_dataset: str) -> str:
    text_fold = TEXT2FOLD.get(pretrain_dataset)
    if text_fold is not None:
        return f"TabSTAR-paper-version-fold-k{text_fold}"

    pretrain_fold = PRETRAIN2FOLD.get(pretrain_dataset)
    if pretrain_fold is not None:
        return f"TabSTAR-eval-320-version-fold-k{pretrain_fold}"

    raise ValueError(f"Unknown dataset: {pretrain_dataset}")