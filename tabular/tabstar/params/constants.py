from enum import StrEnum

from tabular.constants import LORA_BATCH_SIZE

LORA_BATCH = LORA_BATCH_SIZE

E5_SMALL = 'intfloat/e5-small-v2'
D_MODEL = 384
E5_LAYERS = 12

# TODO: this should be part of a branch of experiments.
class NumberVerbalization(StrEnum):
    NONE = 'none'
    RANGE = 'range'
    FULL = 'full'
