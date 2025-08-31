from transformers import PretrainedConfig

from tabstar.arch.config import D_MODEL
from tabstar_paper.pretraining.hyperparameters import TABULAR_LAYERS, TEXTUAL_UNFREEZE_LAYERS


class TabStarConfig(PretrainedConfig):
    model_type = "tabstar"

    def __init__(
        self,
        num_layers: int = TABULAR_LAYERS,
        unfreeze_layers: int = TEXTUAL_UNFREEZE_LAYERS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.unfreeze_layers = unfreeze_layers
        self.d_model = D_MODEL
