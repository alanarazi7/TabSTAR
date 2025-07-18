from datetime import datetime

def get_now() -> str:
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')