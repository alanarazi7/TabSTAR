from dataclasses import dataclass
from typing import Set

from pandas import Series
from sklearn.preprocessing import LabelEncoder

from tabular.preprocessing.nulls import MISSING_VALUE


@dataclass
class ColumnLabelEncoder:
    src: Series
    encoder: LabelEncoder
    train_values: Set[str]


def fit_encode_categorical(s: Series) -> ColumnLabelEncoder:
    encoder = LabelEncoder()
    train_values = set(s).union({MISSING_VALUE})
    encoder.fit(list(train_values))
    return ColumnLabelEncoder(src=s, encoder=encoder, train_values=train_values)

def transform_encoder_categorical(s: Series, encoder: ColumnLabelEncoder) -> Series:
    s = s.apply(lambda v: v if v in encoder.train_values else MISSING_VALUE)
    return encoder.encoder.transform(s).astype(int)