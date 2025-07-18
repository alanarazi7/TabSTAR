def is_numerical(v: float | str | int) -> bool:
    if isinstance(v, str):
        return v.isdigit()
    elif isinstance(v, (int, float)):
        return True
    raise ValueError(f"Unexpected type {type(v)}: {v}")