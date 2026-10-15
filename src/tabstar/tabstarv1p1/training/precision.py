import torch
from torch import GradScaler, autocast, Tensor
from torch.nn import Module
from torch.optim import Optimizer

from tabstar.tabstarv1p1.training.devices import get_precision


class PrecisionManager:

    def __init__(self, device: torch.device):
        self.device = device
        self.use_amp = bool(self.device.type == "cuda")
        self.dtype = get_precision(device=self.device)
        self.use_scaler = self.use_amp and self.dtype == torch.float16
        self.scaler = GradScaler() if self.use_scaler else None

    def autocast_context(self) -> autocast:
        """
        Returns a context manager that configures autocast based on the precision settings.
        """
        return autocast(enabled=self.use_amp, device_type=self.device.type, dtype=self.dtype)

    def backward_loss(self, loss_for_backward: Tensor):
        if not self.use_scaler:
            loss_for_backward.backward()
        else:
            self.scaler.scale(loss_for_backward).backward()

    def update_model(self, model: Module, optimizer: Optimizer):
        if self.use_scaler:
            self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if self.use_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)


