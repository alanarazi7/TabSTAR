from subprocess import Popen, PIPE
from typing import Optional

import torch


def get_device(device: Optional[str] = None) -> torch.device:
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
        return _get_free_gpu()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        return "mps"
    print(f"⚠️ No GPU available, using CPU. This may lead to slow performance.")
    return "cpu"

def _get_free_gpu() -> str:
    gpu_output = Popen(["nvidia-smi", "-q", "-d", "PIDS"], stdout=PIPE, encoding="utf-8")
    gpu_processes = Popen(["grep", "Processes"], stdin=gpu_output.stdout, stdout=PIPE, encoding="utf-8")
    gpu_output.stdout.close()
    processes_output = gpu_processes.communicate()[0]
    for i, line in enumerate(processes_output.strip().split("\n")):
        if line.endswith("None"):
            print(f"Found Free GPU ID: {i}")
            cuda_device = f"cuda:{i}"
            return cuda_device
    raise RuntimeError("No free GPU found")


def get_gpu_num(device: str) -> int:
    prefix = "cuda:"
    assert device.startswith(prefix), f"Device {device} should start with {prefix}"
    return int(device.replace(prefix, ''))