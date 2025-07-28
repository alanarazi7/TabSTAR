import os, time, threading, psutil
from typing import Optional
import torch


class PeakMemoryTracker:
    def __init__(self, phase: str, device: Optional[torch.device] = None, interval: float = 0.5):
        self.phase = phase
        self.device = device
        self.interval = interval
        self.max_rss = 0
        self.max_gpu_used = 0
        self._stop = threading.Event()
        self.gpu_enabled = False
        if device is not None and device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                self.gpu_enabled = True
            except Exception as e:
                print(f"Warning: NVML init failed, disabling GPU tracking: {e}")

    def _sample(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss
                self.max_rss = max(self.max_rss, rss)
            except psutil.NoSuchProcess:
                break
            if self.gpu_enabled:
                try:
                    info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.max_gpu_used = max(self.max_gpu_used, info.used)
                except Exception:
                    pass
            time.sleep(self.interval)

    def __enter__(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self.thread.join(timeout=self.interval * 2)
        self.end_time = time.time()

    def summary(self):
        peak_ram_gb = byte_to_gb(self.max_rss)
        peak_gpu_ram_gb = byte_to_gb(self.max_gpu_used)
        return {
            f'{self.phase}_wall_time_s': self.end_time - self.start_time,
            f'{self.phase}_peak_cpu_gb': peak_ram_gb,
            f'{self.phase}_peak_gpu_gb': peak_gpu_ram_gb,
        }


def byte_to_gb(byte_size: int) -> float:
    return byte_size / (1024 ** 3)