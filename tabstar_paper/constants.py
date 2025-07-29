import os

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
PRETRAIN_BATCH_SIZE = int(os.getenv("PRETRAIN_BATCH_SIZE", 32))
GPU = os.getenv("GPU")
CPU = os.getenv("CPU") is not None

DEVICE = None
if GPU is not None:
    DEVICE = f"cuda:{GPU}"
if CPU:
    DEVICE = "cpu"