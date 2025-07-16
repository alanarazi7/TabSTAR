from transformers import PretrainedConfig


D_MODEL = 384
E5_SMALL = 'intfloat/e5-small-v2'

GLOBAL_BATCH_SIZE = 128
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001

class TabStarConfig(PretrainedConfig):
    model_type = "tabstar"

    def __init__(
        self,
        weight_decay: float = WEIGHT_DECAY,
        macro_batch_size: int = GLOBAL_BATCH_SIZE,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay
        self.macro_batch_size = macro_batch_size
        self.batch_size = batch_size
        assert self.batch_size <= self.macro_batch_size, "Batch size cannot be larger than macro batch size"

    @property
    def accumulation_steps(self) -> int:
        accumulation_steps = self.macro_batch_size // self.batch_size
        assert accumulation_steps * self.batch_size == self.macro_batch_size
        return accumulation_steps
