from dataclasses import dataclass
from typing import List, Dict, Tuple

from torch import nn
from torch.nn.parameter import Parameter

from tabular.tabstar.params.config import TabStarConfig


@dataclass
class ParamGroup:
    params: List[Tuple[str, Parameter]]
    lr: float
    wd: float
    name: str

    def to_dict(self) -> Dict:
        return {"params": [p for _, p in self.params], "lr": self.lr, "weight_decay": self.wd, "name": self.name}

    def split_weights_and_biases(self) -> List[Dict]:
        weights = [p for name, p in self.params if "bias" not in name]
        biases = [p for name, p in self.params if "bias" in name]
        weight_params = ParamGroup(params=[("w", w) for w in weights], lr=self.lr, wd=self.wd, name=f"{self.name}_w")
        bias_params = ParamGroup(params=[("b", b) for b in biases], lr=self.lr, wd=0, name=f"{self.name}_b")
        return [weight_params.to_dict(), bias_params.to_dict()]


def get_groups_for_optimizer(model: nn.Module, config: TabStarConfig) -> List[Dict]:
    # TODO: this function is a bit too complex for no reason due to legacy, improve.
    groups = []
    named_params = [(name, param) for name, param in model.named_parameters()]
    group = ParamGroup(params=named_params, lr=config.lr, wd=config.weight_decay, name="tabstar")
    groups.extend(group.split_weights_and_biases())
    return groups