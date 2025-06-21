from enum import StrEnum

from tabular.constants import LORA_BATCH_SIZE

LORA_LR = 0.001
LORA_BATCH = LORA_BATCH_SIZE
LORA_R = 32

E5_SMALL = 'intfloat/e5-small-v2'
D_MODEL = 384
E5_LAYERS = 12

class NumberVerbalization(StrEnum):
    NONE = 'none'
    RANGE = 'range'
    FULL = 'full'
