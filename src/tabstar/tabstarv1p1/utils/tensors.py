import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        return x.cpu().numpy()
    return np.asarray(x)