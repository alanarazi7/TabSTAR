from typing import Dict

from gpu_tracker.tracker import ResourceUsage

def get_profiling_dict(ru: ResourceUsage) -> Dict[str, float]:
    return {
        # Wall‑clock execution time
        'wall_time': ru.compute_time.time,
        # Peak system RAM usage during profiling
        'peak_ram_system': ru.max_ram.system,
        # Combined RSS total (process + descendants)
        'rss_total_combined': ru.max_ram.combined.total_rss,
        # CPU utilization: main process, peak and average
        'cpu_peak_percent': ru.cpu_utilization.main.max_hardware_percent,
        'cpu_mean_percent': ru.cpu_utilization.main.mean_hardware_percent,
        # GPU metrics (if GPU tracked; zeros otherwise)
        'gpu_peak_ram': ru.max_gpu_ram.system,
        'gpu_peak_util_percent': ru.gpu_utilization.gpu_percentages.max_hardware_percent,
    }