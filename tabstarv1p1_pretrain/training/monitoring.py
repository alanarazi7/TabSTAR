import torch
from torch.nn import Module


def log_grad_stats(model: Module, step: str):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if torch.isnan(grad).any():
            print(f"[BAD GRADIENT] {name} at step {step}: NaN detected")
        if torch.isinf(grad).any():
            print(f"[BAD GRADIENT] {name} at step {step}: Inf detected")
        if grad.abs().max() > 1e5:
            print(f"[BAD GRADIENT] {name} at step {step}: Large gradient {grad.abs().max()}")


def log_param_stats(model: Module, step: str):
    for name, param in model.named_parameters():
        data = param.data
        if torch.isnan(data).any():
            print(f"[BAD PARAM] {name} at step {step}: NaN detected")
        if torch.isinf(data).any():
            print(f"[BAD PARAM] {name} at step {step}: Inf detected")
        if data.abs().max() > 1e5:
            print(f"[BAD PARAM] {name} at step {step}: Large gradient {data.abs().max()}")
