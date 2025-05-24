import os
from enum import StrEnum

import wandb
from dotenv import load_dotenv

from tabular.utils.utils import hash_str


load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

WANDB_PROJECT = "tabular"
WANDB_ENTITY = "tabular_data"

LOG_SEP = "__"


class RunType(StrEnum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    BASELINE = "baseline"


def wandb_run(exp_name: str, run_type: RunType) -> None:
    if WANDB_API_KEY is None:
        print("⚠️ WANDB_API_KEY not found in your .env file! Won't log to wandb.")
        mode = "disabled"
    else:
        mode = "online"
    tags = []
    if run_type == RunType.FINETUNE:
        assert exp_name.count('/') == 1, f"Expected format: pretrain_exp/finetune_exp, got {exp_name}"
        pretrain_exp, finetune_exp = exp_name.split('/')
        tags.append(hash_str(pretrain_exp))
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, reinit=True, name=exp_name, group=str(run_type.value),
               tags=tags, mode=mode)
