import logging
import subprocess
from functools import wraps
import inspect

import wandb
from wandb.errors import CommError

from tabstar_paper.constants import WANDB_API_KEY, WANDB_ENTITY


# TODO: temporary solution, to be rewritten
def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        log_lines = [f"Calling {func.__qualname__} with arguments:"]
        for name, value in bound.arguments.items():
            log_lines.append(f"    {name}: {value!r}")
        logger.info("\n\n".join(log_lines))
        return func(*args, **kwargs)
    return wrapper

def log_all_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_calls(attr_value))
    return cls


def get_current_commit_hash() -> str:
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash[:7]
    except subprocess.CalledProcessError:
        print(f"🆔 Could not get the current commit hash.")
        return ""


def wandb_run(exp_name: str, project: str) -> None:
    mode = "online"
    if WANDB_API_KEY is None:
        print("⚠️ `WANDB_API_KEY` not found in your .env file! Won't log to wandb.")
        mode = "disabled"
    if WANDB_API_KEY and WANDB_ENTITY is None:
        raise ValueError("WANDB_ENTITY is not set! Please set it in your .env file.")
    wandb.login(key=WANDB_API_KEY)
    try:
        wandb.init(entity=WANDB_ENTITY, project=project, reinit=True, name=exp_name, mode=mode)
    except CommError as e:
        print(f"⚠️ WandB couldn't connect to entity `{WANDB_ENTITY}`!")
        raise e
