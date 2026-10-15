GLOBAL_BATCH = 128


def set_accumulation_steps(batch_size: int, global_batch: int):
    if global_batch % batch_size != 0:
        raise ValueError("Global batch size must be divisible by local batch size.")
    if global_batch < batch_size:
        raise ValueError("Global batch size must be greater than or equal to local batch size.")
    return global_batch // batch_size
