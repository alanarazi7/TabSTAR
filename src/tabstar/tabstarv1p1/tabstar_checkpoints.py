from typing import Optional

DEFAULT_CLS_CHECKPOINT = "alana89/TabSTAR-v1.1-cls-20261011"
DEFAULT_REG_CHECKPOINT = "alana89/TabSTAR-v1.1-reg-20261009"


def get_tabstar_version(pretrain_dataset_or_path: Optional[str] = None, is_cls: Optional[bool] = None) -> str:
    if pretrain_dataset_or_path is not None:
        if isinstance(pretrain_dataset_or_path, str):
            return pretrain_dataset_or_path
        raise ValueError(f"Unknown pretrain_dataset_or_path: {pretrain_dataset_or_path}")
    if is_cls is None:
        raise ValueError("is_cls is required to resolve a default checkpoint")
    return DEFAULT_CLS_CHECKPOINT if is_cls else DEFAULT_REG_CHECKPOINT
