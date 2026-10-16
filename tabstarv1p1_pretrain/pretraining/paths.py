from os.path import join

# Sibling of the checkout, shared by every worktree.
PRETRAIN_MODELS_DIR = "../tabstar_pretrain_models"


def pretrain_exp_dir(exp_name: str) -> str:
    return join(PRETRAIN_MODELS_DIR, exp_name)

def pretrain_args_path(exp_name: str) -> str:
    return join(pretrain_exp_dir(exp_name), "pretrain_args.json")


def get_pretrain_checkpoint(run_name: str, epoch: int) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, f"pretrain_checkpoint_{epoch}")

def get_finetunability_snapshot_path(run_name: str, epoch: int) -> str:
    """A transient per-epoch snapshot used to score that epoch's finetunability signal, then
    deleted — unlike the persisted training checkpoints.
    """
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, f"finetunability_snapshot_{epoch}")