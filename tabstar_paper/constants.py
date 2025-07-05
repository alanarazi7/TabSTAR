import os

PRETRAIN_BATCH_SIZE = int(os.getenv("PRETRAIN_BATCH_SIZE", 32))
GPU = os.getenv("GPU")
if GPU is not None:
    GPU = f"cuda:{GPU}"