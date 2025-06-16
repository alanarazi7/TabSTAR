from typing import Optional

import torch


def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = _get_device_type()
    return torch.device(device)

def clear_cuda_cache():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

def _get_device_type() -> str:
    if torch.cuda.is_available():
        clear_cuda_cache()
        try:
            best = _get_most_free_gpu()
            if best is not None:
                return best
        except Exception:
            return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return "cpu"

def _get_most_free_gpu() -> Optional[str]:
    best_idx = None
    best_free_mem = 0
    for idx in range(torch.cuda.device_count()):
        free_mem, _ = torch.cuda.mem_get_info(idx)
        if free_mem > best_free_mem:
            best_free_mem = free_mem
            best_idx = f'cuda:{idx}'
    return best_idx
