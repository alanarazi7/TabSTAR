from optuna.terminator.improvement.emmr import torch


def get_device() -> torch.device:
    return torch.device(_get_device_type())

def _get_device_type() -> str:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    return "cpu"
