import time
from datetime import datetime

def get_now() -> str:
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def elapsed_minutes(start: float) -> float:
    return (time.time() - start) / 60

def remove_time_from_name(s: str) -> str:
    while s[0] in '0123456789_':
        s = s[1:]
    return s