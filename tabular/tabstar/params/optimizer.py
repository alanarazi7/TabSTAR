from typing import List, Tuple
from torch.nn.parameter import Parameter

from tabular.tabstar.arch.arch import TabStarModel

NAMED_PARAMETERS = List[Tuple[str, Parameter]]

def get_tabstar_parameters_by_group(model: TabStarModel) -> Tuple[NAMED_PARAMETERS, NAMED_PARAMETERS]:
    # TODO: This is not relevant anymore, we don't use differential LR
    text_params = []
    tabular_params = []

    for name, param in model.named_parameters():
        if name.startswith("text_encoder"):
            text_params.append((name, param))
        else:
            tabular_params.append((name, param))

    return text_params, tabular_params