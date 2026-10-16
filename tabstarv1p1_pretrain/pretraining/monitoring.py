import subprocess
from typing import Dict, Optional

import torch

from tabstarv1p1_pretrain.utils.hardware import byte_to_gb


def print_gpu_mem_gb(device: torch.device):
    mem = gpu_mem_gb(device)
    if mem == 0:
        return ""
    else:
        return f"GPU {mem:.2f}GB"


def gpu_mem_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    reserved = torch.cuda.memory_reserved(device.index)
    reserved = byte_to_gb(reserved)
    return reserved


def reset_peak_gpu_mem(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index)


def peak_gpu_mem_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return byte_to_gb(torch.cuda.max_memory_reserved(device.index))


def gpu_contention(device: torch.device) -> Optional[Dict[str, float]]:
    """Whole-device (every process on it, not just ours) utilization and memory for `device`'s
    GPU index, via one `nvidia-smi` query. A rough contention signal on a shared server: if
    `mem_used_gb` dwarfs our own `gpu_mem_gb`, something else is sharing the card. None if
    unavailable (no CUDA device, or `nvidia-smi` missing/failed).
    """
    if device.type != "cuda" or device.index is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={device.index}",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        util_pct, mem_used_mb, mem_total_mb = (float(x) for x in out.stdout.strip().split(","))
        return {"util_pct": util_pct, "mem_used_gb": mem_used_mb / 1024, "mem_total_gb": mem_total_mb / 1024}
    except Exception:
        return None