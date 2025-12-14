MAX_EPOCHS = 50
FINETUNE_PATIENCE = 5
LORA_LR = 0.001
LORA_WD = 0.0
LORA_DROPOUT = 0.1
LORA_R = 32
LORA_ALPHA = 2
LORA_BATCH = 64
GLOBAL_BATCH = 128
VAL_BATCH = 128


def set_accumulation_steps(batch_size: int, global_batch: int):
    if global_batch % batch_size != 0:
        raise ValueError("Global batch size must be divisible by local batch size.")
    if global_batch < batch_size:
        raise ValueError("Global batch size must be greater than or equal to local batch size.")
    return global_batch // batch_size