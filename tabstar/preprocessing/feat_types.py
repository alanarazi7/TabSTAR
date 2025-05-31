from dataclasses import dataclass, field
from typing import Set, Any, List, Self

from pandas import DataFrame, Series

from tabstar.preprocessing.nulls import get_valid_values


# MIN_TEXT_UNIQUE_RATIO = 0.8
# MIN_TEXT_UNIQUE_FREQUENCY = 100
# MAX_NUMERIC_FOR_CATEGORICAL = 50
# MIN_NUMERIC_UNIQUE = 10

# set_option('future.no_silent_downcasting', True)

# @dataclass
# class ValueStats:
#     name: str
#     values: List[Any]
#     unique: List[Any]
#     n_unique: int
#     unique_ratio: float
#
#     @classmethod
#     def from_values(cls, series: Series) -> Self:
#         name = str(series.name)
#         values = get_valid_values(series)
#         unique = list(set(values))
#         n_unique = len(unique)
#         unique_ratio = 0 if not len(values) else n_unique / len(values)
#         # TODO: add missing ratio information?
#         return cls(name=name, values=values, unique=unique, n_unique=n_unique, unique_ratio=unique_ratio)
#
#     @property
#     def all_numerical(self) -> bool:
#         return all(is_numerical(v) for v in self.values)
#
#     @property
#     def all_numerical_but_one(self):
#         are_numerical = {v for v in self.values if is_numerical(v)}
#         return len(set(are_numerical)) == self.n_unique - 1




DATE_TYPES = {'datetime64[ns]'}
NUMERIC_TYPES = {'int64', 'int', 'float64', 'float'}
SEMANTIC_TYPES = {'object', 'bool'}
SUPPORTED_TYPES = DATE_TYPES | NUMERIC_TYPES | SEMANTIC_TYPES

@classmethod
class FeatureTypes:
    numerical: Set[str] = field(default_factory=set)
    semantic: Set[str] = field(default_factory=set)
    dates: Set[str] = field(default_factory=set)


# TODO: for future versions, it's best if we try to rely on logics of skrub or similar packages
def get_feature_types(x: DataFrame) -> FeatureTypes:
    feature_types = FeatureTypes()
    for col, dtype in x.dtypes.items():
        col_name = str(col)
        if dtype in DATE_TYPES:
            feature_types.dates.add(col_name)
        elif dtype in NUMERIC_TYPES:
            assert False, "here need to check if we can convert to numerical with tricks, see `_deduce_feature_type`"
            # if stats.n_unique > MAX_NUMERIC_FOR_CATEGORICAL:
            #     if stats.all_numerical or stats.all_numerical_but_one:
            #         verbose_print(f"📊📊📊 Categorical with too many numerical values, converting automatically: {stats}")
            #         return FeatureType.NUMERIC
            feature_types.numerical.add(col_name)
        elif dtype in SEMANTIC_TYPES:
            feature_types.semantic.add(col_name)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}. Convert to any of: {SUPPORTED_TYPES=}")


def is_mostly_numerical(y: Series) -> bool:
    raise NotImplementedError()


def is_numerical(v: float | str | int) -> bool:
    if isinstance(v, str):
        return v.isdigit()
    elif isinstance(v, (int, float)):
        return True
    raise ValueError(f"Unexpected type {type(v)}: {v}")