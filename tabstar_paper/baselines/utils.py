import logging
from functools import wraps
import inspect

def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        log_lines = [f"Calling {func.__qualname__} with arguments:"]
        for name, value in bound.arguments.items():
            log_lines.append(f"    {name}: {value!r}")
        logger.info("\n\n".join(log_lines))
        return func(*args, **kwargs)
    return wrapper

def log_all_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_calls(attr_value))
    return cls
