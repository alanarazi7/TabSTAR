import subprocess
from typing import Dict, List, Optional

import wandb
from wandb.errors import CommError

from tabstarv1p1_pretrain.constants import WANDB_API_KEY, WANDB_ENTITY


def git_info() -> str:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
    return f"{branch}@{commit}"


def assert_same_git_hash(git: str):
    src_hash = git[:7]
    curr_hash = get_current_commit_hash()
    assert src_hash == curr_hash, f"Source {src_hash=} does not match current {curr_hash=}"


def get_current_commit_hash() -> str:
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash[:7]
    except subprocess.CalledProcessError:
        print(f"🆔 Could not get the current commit hash.")
        return ""


def wandb_run(exp_name: str, project: str, tags: Optional[List[str]] = None) -> None:
    mode = "online"
    if WANDB_API_KEY is None:
        print("⚠️ `WANDB_API_KEY` env var not set! Won't log to wandb.")
        mode = "disabled"
    elif WANDB_ENTITY is None:
        raise ValueError("`WANDB_ENTITY` env var is not set!")
    else:
        wandb.login(key=WANDB_API_KEY)
    try:
        wandb.init(entity=WANDB_ENTITY, project=project, reinit=True, name=exp_name, mode=mode, tags=tags)
    except CommError as e:
        print(f"⚠️ WandB couldn't connect to entity `{WANDB_ENTITY}`!")
        raise e


def wandb_finish(d_summary: Dict):
    wandb.log(d_summary)
    wandb.summary.update(d_summary)
    wandb.finish()
    print(f"Summary logged to WandB: {d_summary}")