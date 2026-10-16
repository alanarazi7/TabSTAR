import os

from dotenv import load_dotenv

from tabstar.tabstarv1p1.constants import SEED

load_dotenv()

MONITOR = os.getenv("MONITOR") is not None
MEMORY = os.getenv("MEMORY") is not None
CPU = os.getenv("CPU") is not None

DEVICE = None
if os.getenv("GPU") is not None:
    DEVICE = f"cuda:{os.getenv('GPU')}"
if CPU:
    DEVICE = "cpu"

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
