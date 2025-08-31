from torch import nn
from torch.optim import AdamW

from tabstar_paper.pretraining.hyperparameters import TrainingArgs


def get_optimizer(model: nn.Module, args: TrainingArgs) -> AdamW:
    assert isinstance(model, nn.Module)
    biases, weights = [], []
    for name, p in model.named_parameters():
        if "bias" in name:
            biases.append(p)
        else:
            weights.append(p)
    lr = args.learning_rate
    param_groups = [{"params": weights, "weight_decay": args.weight_decay, "lr": lr, "name": "weights"},
                    {"params": biases, "weight_decay": 0.0, "lr": lr, "name": "biases"}]
    optimizer = AdamW(param_groups)
    return optimizer