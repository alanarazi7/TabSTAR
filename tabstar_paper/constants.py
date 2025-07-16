import os

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PRETRAIN_BATCH_SIZE = int(os.getenv("PRETRAIN_BATCH_SIZE", 32))
GPU = os.getenv("GPU")

if GPU is not None:
    GPU = f"cuda:{GPU}"