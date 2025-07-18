import os
from typing import Optional

import torch


def get_device(device: Optional[str | torch.device] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        device = _get_device_type()
    if 'cuda' in device:
        gpu_num = get_gpu_num(device)
        torch.cuda.set_device(gpu_num)
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
        return _get_most_free_gpu()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return "cpu"

def _get_most_free_gpu() -> Optional[str]:
    best_idx = None
    best_free_mem = 0
    for idx in range(torch.cuda.device_count()):
        try:
            free_mem, _ = torch.cuda.mem_get_info(idx)
            if free_mem > best_free_mem:
                best_free_mem = free_mem
                best_idx = f'cuda:{idx}'
        except RuntimeError as e:
            print(f"[WARN] Could not check GPU {idx}: {e}")
            continue
    if best_idx is None:
        raise RuntimeError("No available GPU found: all GPUs are out of memory or inaccessible.")
    return best_idx


def get_gpu_num(device: str) -> int:
    prefix = "cuda:"
    assert device.startswith(prefix), f"Device {device} should start with {prefix}"
    return int(device.replace(prefix, ''))