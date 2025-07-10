import hashlib
from datetime import datetime

from tabular.constants import VERBOSE

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
