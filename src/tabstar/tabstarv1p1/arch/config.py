from transformers import PretrainedConfig

from tabstar.tabstarv1p1.constants import TEXT_BATCH_SIZE
from tabstar.tabstarv1p1.constants import D_MODEL, INTERACTION_LAYERS, TEXT_MODEL, D_NUM, REG_DIST_BINS
from tabstar.tabstarv1p1.constants import TEXTUAL_UNFREEZE_LAYERS


class TabStarConfig(PretrainedConfig):
    model_type = "tabstar"

    def __init__(
        self,
        interaction_layers: int = INTERACTION_LAYERS,
        d_model: int = D_MODEL,
        text_model: str = TEXT_MODEL,
        unfreeze_layers: int = TEXTUAL_UNFREEZE_LAYERS,
        text_batch_size: int = TEXT_BATCH_SIZE,
        d_num: int = D_NUM,
        reg_dist_bins: int = REG_DIST_BINS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interaction_layers = interaction_layers
        self.d_model = d_model
        self.text_model = text_model
        self.unfreeze_layers = unfreeze_layers
        self.text_batch_size = text_batch_size
        self.d_num = d_num
        self.reg_dist_bins = reg_dist_bins
