from enum import StrEnum

from tabular.constants import LORA_BATCH_SIZE

TABULAR_LAYERS = 6
GLOBAL_BATCH_SIZE = 128
WEIGHT_DECAY = 0.001
BASE_LR = 0.00005
TEXTUAL_UNFREEZE_LAYERS = 6

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
