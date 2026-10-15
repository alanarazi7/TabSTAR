from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoTokenizer

from tabstar.tabstarv1p1.constants import TEXT_MODEL

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


def download_tabstar(repo_id: Optional[str] = None):
    """Warm the cache before a fit, so an unreachable checkpoint fails here and not mid-training.
    Without a repo_id, both default checkpoints are fetched."""
    repo_ids = [repo_id] if repo_id is not None else [DEFAULT_CLS_CHECKPOINT, DEFAULT_REG_CHECKPOINT]
    for repo in repo_ids:
        _snapshot(repo)
    # The weights carry no tokenizer of their own; the text encoder's is fetched separately.
    AutoTokenizer.from_pretrained(TEXT_MODEL)


def _snapshot(repo_id: str):
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        snapshot_download(repo_id=repo_id)
