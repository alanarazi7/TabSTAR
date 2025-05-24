from typing import List

from tabular.tabstar.params.constants import E5_LAYERS
from torch import nn

from tabular.utils.utils import cprint


def print_model_summary(model: nn.Module):
    m_total_params = sum(p.numel() for p in model.parameters()) / 1000000
    m_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    cprint(f"Total parameters: {m_total_params:.2f}M. Trainable: {m_trainable:.2f}M")
    for name, submodule in model.named_children():
        submodule_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        print(f"{name}: {submodule_params:,} parameters")
    if hasattr(model, 'text_encoder'):
        for name, submodule in model.text_encoder.named_children():
            submodule_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            total_submodule_params = sum(p.numel() for p in submodule.parameters())
            print(f"Text encoder {name}: {submodule_params:,}/{total_submodule_params:,} trained parameters")

def get_last_layers_num(to_unfreeze: int, total_layers: int = E5_LAYERS) -> List[int]:
    layers_reversed = list(reversed(range(total_layers)))
    unfrozen = layers_reversed[:to_unfreeze]
    return unfrozen