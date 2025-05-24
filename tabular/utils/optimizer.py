from dataclasses import dataclass
from typing import List, Dict, Tuple

from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler

from tabular.tabstar.params.config import TabStarConfig
from tabular.tabstar.params.optimizer import get_tabstar_parameters_by_group

WARMUP_PROPORTION = 0.1
MAX_EPOCHS = 50


def get_optimizer(model: nn.Module, config: TabStarConfig) -> Tuple[AdamW, LRScheduler]:
    if config.is_pretrain:
        params = get_groups_for_optimizer(model=model, config=config)
    else:
        params = [{"params": model.parameters(), "lr": config.lr, "name": "lora_lr"}]
    optimizer = AdamW(params)
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=config.lr, total_steps=MAX_EPOCHS,
                           pct_start=WARMUP_PROPORTION, anneal_strategy='cos')
    return optimizer, scheduler


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
    # TODO: this function was useful when we had differential LR, now it is legacy
    groups = []
    text_params, tab_params = get_tabstar_parameters_by_group(model)
    for grp, lr, name in [(tab_params, config.lr, "tab")]:
        group = ParamGroup(params=grp, lr=lr, wd=config.weight_decay, name=name)
        groups.extend(group.split_weights_and_biases())
    return groups