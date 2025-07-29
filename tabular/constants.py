import os

def get_env_bool(env_var: str) -> bool:
    return os.getenv(env_var, "False").lower() in ("true", "1", "yes")

VERBOSE = get_env_bool("VERBOSE")
OPTUNA_BUDGET = int(os.getenv("OPTUNA_BUDGET", 60 * 60 * 4))