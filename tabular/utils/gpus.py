from subprocess import Popen, PIPE

import torch

from tabular.constants import GPU


def find_best_cuda_gpu() -> str:
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


def get_max_gpu_memory(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    memory_gb = int(properties.total_memory / (1024 ** 3))
    return memory_gb


def get_device() -> str:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if GPU is not None:
                gpu = int(GPU)
                assert gpu in range(torch.cuda.device_count()), f"GPU {gpu} not available"
                return f"cuda:{gpu}"
            return find_best_cuda_gpu()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            return "mps"
        return "cpu"
    except Exception as e:
        raise Exception(f"Could not get a valid device: {e}")

