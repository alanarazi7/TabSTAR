from contextlib import contextmanager

import torch
import wandb


def get_cpu_cores() -> int:
    num_gpus = torch.cuda.device_count()
    num_cpu_cores = torch.get_num_threads()
    num_gpus = max(1, num_gpus)
    cpus_per_gpus = max(1, num_cpu_cores // num_gpus)
    return cpus_per_gpus


def assert_bf16_supported(device: torch.device):
    if device.type != "cuda":
        raise NotImplementedError("Pretraining is only supported on GPU.")
    if not torch.cuda.is_bf16_supported():
        raise NotImplementedError("Pretraining is only supported on GPU with bf16 support.")


@contextmanager
def safe_gpu_training():
    try:
        yield
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Cleaning GPU cache and finishing wandb...")
        try:
            wandb.finish()
        except Exception as e:
            print(f"Exception while finishing wandb: {e}")
        torch.cuda.empty_cache()
        print("GPU cache cleared. Exiting.")
        raise  # re-raise to stop execution
    except Exception as e:
        # Catch all other exceptions to clean GPU memory
        print(f"Exception caught: {e}. Cleaning GPU cache...")
        try:
            wandb.finish()
        except Exception as e2:
            print(f"Exception while finishing wandb: {e2}")
        torch.cuda.empty_cache()
        print("GPU cache cleared. Raising exception.")
        raise