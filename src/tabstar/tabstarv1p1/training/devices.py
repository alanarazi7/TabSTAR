import gc
import os
from subprocess import Popen, PIPE
from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    gpu = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpu is not None:
        print(f"🖥️ Using CUDA_VISIBLE_DEVICES={gpu}. Setting device to 'cuda'.")
        if isinstance(gpu, int) or gpu.isdigit():
            return torch.device("cuda")
        raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES value: {gpu}. It should be an integer for now.")
    if device is None:
        device = _get_device_type()
    return torch.device(device)

def clear_cuda_cache():
    if torch.cuda.is_available():
        # An OOM exception's traceback holds a reference cycle to the tensors live at
        # the point of failure; empty_cache() alone can't reclaim memory still referenced.
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

def get_precision(device: Optional[torch.device] = None) -> torch.dtype:
    # Autocast only runs on CUDA. Anywhere else a reduced dtype would meet float32 weights.
    if device is not None and device.type != "cuda":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

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
            return f"cuda:{i}"
    raise RuntimeError("No free GPU found! However, pass `CUDA_VISIBLE_DEVICES` if you insist on using one.")
