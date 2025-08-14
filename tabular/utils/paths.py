from os.path import join

from tabstar_paper.pretraining.paths import pretrain_exp_dir


def get_model_path(run_name: str) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, "best_checkpoint")

def get_checkpoint(run_name: str, epoch: int) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, f"checkpoint_{epoch}")
