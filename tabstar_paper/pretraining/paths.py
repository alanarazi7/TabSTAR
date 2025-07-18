from os.path import join


def pretrain_exp_dir(exp_name: str) -> str:
    return join(".tabstar_pretrain", exp_name)

def pretrain_args_path(exp_name: str) -> str:
    return join(pretrain_exp_dir(exp_name), "pretrain_args.json")