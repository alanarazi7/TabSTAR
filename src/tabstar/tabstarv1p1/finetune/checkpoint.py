from datetime import datetime
from os import getpid, makedirs
from os.path import abspath, join


def make_run_dir() -> str:
    """Scratch directory for one finetune's saved weights, disambiguated so concurrent runs in the
    same working directory never read or delete each other's files."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{getpid()}"
    run_dir = abspath(join(".tabstar_downstream", run_id))
    makedirs(run_dir, exist_ok=True)
    return run_dir
