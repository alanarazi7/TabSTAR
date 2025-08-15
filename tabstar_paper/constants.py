import os

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
HF_TOKEN = os.getenv("HF_TOKEN")
PRETRAIN_BATCH_SIZE = int(os.getenv("PRETRAIN_BATCH_SIZE", 32))
TIME_BUDGET = int(os.getenv("TIME_BUDGET", 60 * 60 * 4))
GPU = os.getenv("GPU")
CPU = os.getenv("CPU") is not None

DEVICE = None
if GPU is not None:
    DEVICE = f"cuda:{GPU}"
if CPU:
    DEVICE = "cpu"