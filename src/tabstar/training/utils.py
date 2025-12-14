import os
import random
from typing import List

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from torch import Tensor

from tabstar.constants import SEED


TABSTAR_REPO_ID = "alana89/TabSTAR"


def fix_seed(seed: int = SEED):
    if seed is None:
        return
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def concat_predictions(y_pred: List[Tensor]) -> np.ndarray:
    return np.concatenate([p.cpu().detach().numpy() for p in y_pred])


def download_tabstar():
    try:
        snapshot_download(repo_id=TABSTAR_REPO_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        snapshot_download(repo_id=TABSTAR_REPO_ID)