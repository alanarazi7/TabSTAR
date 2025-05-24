import os

def get_env_bool(env_var: str) -> bool:
    return os.getenv(env_var, "False").lower() in ("true", "1", "yes")

VERBOSE = get_env_bool("VERBOSE")
GPU = os.getenv("GPU", None)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LORA_BATCH_SIZE = int(os.getenv("LORA_BATCH_SIZE", 64))
OPTUNA_CPU = int(os.getenv("OPTUNA_CPU", 8))
OPTUNA_BUDGET = int(os.getenv("OPTUNA_BUDGET", 60 * 60 * 4))

NUM_VERBALIZATION = os.getenv("NUM_VERBALIZATION", "full")
assert NUM_VERBALIZATION in {'none', 'range', 'full'}