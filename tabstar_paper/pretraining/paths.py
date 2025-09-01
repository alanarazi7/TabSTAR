from os.path import join


def pretrain_exp_dir(exp_name: str) -> str:
    return join(".tabstar_pretrain", exp_name)

def pretrain_args_path(exp_name: str) -> str:
    return join(pretrain_exp_dir(exp_name), "pretrain_args.json")


def get_model_path(run_name: str) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, "best_checkpoint")

def get_checkpoint(run_name: str, epoch: int) -> str:
    main_dir = pretrain_exp_dir(run_name)
    return join(main_dir, f"checkpoint_{epoch}")