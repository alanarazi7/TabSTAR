from pandas import Series


def raise_if_null_target(y: Series):
    missing = y.isnull()
    y_missing = missing.sum()
    if y_missing > 0:
        raise ValueError(f"Target variable {y.name} has {y_missing} null values, please handle them before training.")