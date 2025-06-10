import torch


def get_device() -> torch.device:
    return torch.device(_get_device_type())

def clear_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

def _get_device_type() -> str:
    if torch.cuda.is_available():
        clear_cuda_cache()
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return "cpu"
