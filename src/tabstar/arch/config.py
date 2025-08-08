from transformers import PretrainedConfig


D_MODEL = 384
E5_SMALL = 'intfloat/e5-small-v2'
WEIGHT_DECAY = 0.001

# TODO: this actually should include many arch parameters: /tabular/tabstar/params/config.py
class TabStarConfig(PretrainedConfig):
    model_type = "tabstar"

    def __init__(
        self,
        d_model: int = D_MODEL,
        embedding_model: str = E5_SMALL,
        weight_decay: float = WEIGHT_DECAY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding_model = embedding_model
        self.weight_decay = weight_decay
