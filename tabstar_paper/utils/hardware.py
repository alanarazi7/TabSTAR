import psutil
from typing import Dict, Optional
import torch

def get_hardware_dict(device: torch.device) -> Dict:
    return {**_get_gpu_dict(device), **_get_cpu_dict()}


def _get_gpu_dict(device: torch.device) -> Dict:
    use_gpu = False
    gpu_type = None
    gpu_total_mem_gb = None
    if device.type == 'cuda':
        use_gpu = True
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_type = torch.cuda.get_device_name(gpu_index)
        gpu_total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
        gpu_total_mem_gb = byte_to_gb(gpu_total_mem)
    return {"gpu_type": gpu_type, "gpu_total_memory_gb": gpu_total_mem_gb, "use_gpu": use_gpu}


def _get_cpu_dict() -> Dict:
    cpu_cores = torch.get_num_threads()
    ram_bytes = psutil.virtual_memory().total
    ram_gb = byte_to_gb(ram_bytes)
    cpu_name = get_cpu_name_linux()
    return {"cpu_cores": cpu_cores, "system_ram_gb": ram_gb, "cpu_name": cpu_name}


def byte_to_gb(byte_size: int) -> float:
    return byte_size / (1024 ** 3)


def get_cpu_name_linux() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
