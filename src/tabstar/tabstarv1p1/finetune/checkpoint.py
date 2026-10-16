from datetime import datetime
from os import getpid, makedirs
from os.path import abspath, join
from typing import Optional

DOWNSTREAM_DIR = ".tabstar_downstream"


def make_run_dir(output_dir: Optional[str] = None) -> str:
    """Scratch directory for one finetune's saved weights, disambiguated so concurrent runs in the
    same working directory never read or delete each other's files."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{getpid()}"
    run_dir = abspath(join(output_dir or DOWNSTREAM_DIR, run_id))
    makedirs(run_dir, exist_ok=True)
    return run_dir
