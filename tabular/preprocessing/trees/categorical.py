from dataclasses import dataclass
from typing import List, Set

from pandas import Series
from sklearn.preprocessing import LabelEncoder

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.nulls import MISSING_VALUE
from tabular.preprocessing.splits import DataSplit, get_x_train
from tabular.utils.utils import verbose_print


@dataclass
class ColumnLabelEncoder:
    src: Series
    encoder: LabelEncoder
    train_values: Set[str]


def process_x_cat_to_indices(raw: RawDataset, splits: List[DataSplit]):
    """Iterate feature-by-feature and encode all the values per feature, 'missing' included."""
    x_train = get_x_train(x=raw.x, splits=splits)
    verbose_print(f"🐈 Encoding {len(raw.bool_cat)} categorical features for {raw.sid} dataset: {raw.bool_cat}")
    for col in raw.bool_cat:
        col_encoder = fit_encode_categorical(s=x_train[col])
        raw.x[col] = transform_encoder_categorical(s=raw.x[col], encoder=col_encoder)


def fit_encode_categorical(s: Series) -> ColumnLabelEncoder:
    encoder = LabelEncoder()
    train_values = set(s).union({MISSING_VALUE})
    encoder.fit(list(train_values))
    return ColumnLabelEncoder(src=s, encoder=encoder, train_values=train_values)

def transform_encoder_categorical(s: Series, encoder: ColumnLabelEncoder) -> Series:
    s = s.apply(lambda v: v if v in encoder.train_values else MISSING_VALUE)
    return encoder.encoder.transform(s).astype(int)