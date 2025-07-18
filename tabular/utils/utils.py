from datetime import datetime

from tabular.constants import VERBOSE

def get_now() -> str:
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def verbose_print(s: str):
    # TODO: use logger
    if VERBOSE:
        print(s)
