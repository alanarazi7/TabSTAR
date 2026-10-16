from typing import Set, Dict

from pandas import DataFrame


def fit_numerical_median(x: DataFrame, numerical_features: Set[str]) -> Dict[str, float]:
    return {col: x[col].median() for col in numerical_features}

def transform_numerical_features(x: DataFrame, numerical_medians: Dict[str, float]) -> DataFrame:
    x = x.copy()
    for col, train_median in numerical_medians.items():
        x[col] = x[col].fillna(train_median)
    return x