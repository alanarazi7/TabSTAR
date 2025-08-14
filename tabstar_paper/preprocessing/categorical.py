from typing import Set, Dict

from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder

from tabstar.preprocessing.nulls import MISSING_VALUE


def fit_categorical_encoders(x: DataFrame, categorical_features: Set[str]) -> Dict[str, LabelEncoder]:
    return {col: fit_encode_categorical(s=x[col]) for col in categorical_features}

def fit_encode_categorical(s: Series) -> LabelEncoder:
    encoder = LabelEncoder()
    train_values = set(s).union({MISSING_VALUE})
    encoder.fit(list(train_values))
    return encoder

def transform_categorical_features(x: DataFrame, categorical_encoders: Dict[str, LabelEncoder]) -> DataFrame:
    x = x.copy()
    for col, encoder in categorical_encoders.items():
        s = x[col].apply(lambda v: v if v in encoder.classes_ else MISSING_VALUE)
        s = encoder.transform(s).astype(int)
        x[col] = s
    return x