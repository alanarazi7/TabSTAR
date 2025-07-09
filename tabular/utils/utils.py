import hashlib
import os
import random
from datetime import datetime

import numpy as np
import torch

from tabular.constants import VERBOSE

SEED = 42

def fix_seed(seed: int = SEED):
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


def get_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_today() -> str:
    return datetime.now().strftime('%Y_%m_%d')

def verbose_print(s: str):
    # TODO: use logger
    if VERBOSE:
        print(s)

def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]
