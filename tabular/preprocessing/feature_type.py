from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set, Any, List, Self, Optional

from pandas import DataFrame, Series, set_option

from tabular.datasets.manual_curation_obj import CuratedDataset, CuratedFeature
from tabular.preprocessing.dates import series_to_dt
from tabular.preprocessing.nulls import get_valid_values, MISSING_VALUE, convert_series_to_numeric
from tabular.preprocessing.objects import FeatureType, FEAT2EMOJI
from tabular.tabstar.preprocessing.numerical_utils import is_numerical
from tabular.utils.utils import verbose_print

MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100
MAX_NUMERIC_FOR_CATEGORICAL = 50
MIN_NUMERIC_UNIQUE = 10


set_option('future.no_silent_downcasting', True)

@dataclass
class ValueStats:
    name: str
    values: List[Any]
    unique: List[Any]
    n_unique: int
    unique_ratio: float

    @classmethod
    def from_values(cls, series: Series) -> Self:
        name = str(series.name)
        values = get_valid_values(series)
        unique = list(set(values))
        n_unique = len(unique)
        unique_ratio = 0 if not len(values) else n_unique / len(values)
        # TODO: add missing ratio information?
        return cls(name=name, values=values, unique=unique, n_unique=n_unique, unique_ratio=unique_ratio)

    @property
    def all_numerical(self) -> bool:
        return all(is_numerical(v) for v in self.values)

    @property
    def all_numerical_but_one(self):
        are_numerical = {v for v in self.values if is_numerical(v)}
        return len(set(are_numerical)) == self.n_unique - 1

    @property
    def non_numerical(self) -> List[Any]:
        return [v for v in self.values if not is_numerical(v)]

    @property
    def common(self) -> List[Any]:
        cnt = Counter(self.values)
        most_common = [v for v, _ in cnt.most_common(20)]
        return most_common

    def __repr__(self) -> str:
        unique_str = f"Unique: {self.n_unique}"
        if self.n_unique > 2:
            unique_str += f" ({self.unique_ratio:.1%})"
        return f"{self.name} | {unique_str} | {self.common}"


def convert_dtypes(x: DataFrame, feature_types: Dict[FeatureType, Set[str]], curation: CuratedDataset):
    for feat_type, cols in feature_types.items():
        for col in cols:
            feat = curation.get_feature(col)
            if feat and feat.processing_func is not None:
                x[col] = x[col].apply(feat.processing_func)
            if feat_type == FeatureType.NUMERIC:
                missing_value = feat.numeric_missing if feat else None
                x[col] = convert_series_to_numeric(x[col], missing_value=missing_value)
            elif feat_type in {FeatureType.CATEGORICAL, FeatureType.TEXT, FeatureType.BOOLEAN}:
                if feat_type == FeatureType.BOOLEAN:
                    assert len(set(get_valid_values(x[col]))) == 2
                x[col] = x[col].astype(object).fillna(MISSING_VALUE).astype(str)
            elif feat_type == FeatureType.DATE:
                x[col] = series_to_dt(x[col])
            else:
                raise ValueError(f"Unsupported feature type: {feat_type}")


def get_feature_types(x: DataFrame, curation: CuratedDataset, feat_types: Dict[str, FeatureType]) -> Dict[FeatureType, Set[str]]:
    feature_types = {f: set() for f in FeatureType}
    for col in x.columns:
        stats = ValueStats.from_values(series=x[col])
        curated = curation.get_feature(col)
        assumed_type = feat_types[col]
        feat_type = _deduce_feature_type(stats=stats, assumed_type=assumed_type, curated=curated)
        verbose_print(f"{FEAT2EMOJI[feat_type]} Feature {feat_type} | {stats}")
        feature_types[feat_type].add(col)
    unsupported = list(feature_types.pop(FeatureType.UNSUPPORTED))
    x.drop(columns=unsupported, inplace=True)
    return feature_types

def _deduce_feature_type(stats: ValueStats, assumed_type: FeatureType, curated: Optional[CuratedFeature]) -> FeatureType:
    """Detecting feature type based on curation, OpenML metadata and automatic heuristics."""
    if curated and curated.feat_type:
        return curated.feat_type

    if stats.n_unique <= 1:
        return FeatureType.UNSUPPORTED
    if stats.n_unique == 2:
        return FeatureType.BOOLEAN

    if assumed_type == FeatureType.DATE:
        return assumed_type
    elif assumed_type == FeatureType.NUMERIC:
        if curated and curated.value_mapping:
            # The user can be minimalist and provide a mapping, this automatically turns into a non-numeric feature
            return FeatureType.CATEGORICAL
        if stats.n_unique < MIN_NUMERIC_UNIQUE:
            verbose_print(f"📊📊📊 Numeric with too less: {stats}")
        return FeatureType.NUMERIC
    elif assumed_type in {FeatureType.CATEGORICAL, FeatureType.TEXT}:
        if stats.n_unique > MAX_NUMERIC_FOR_CATEGORICAL:
            if stats.all_numerical or stats.all_numerical_but_one:
                verbose_print(f"📊📊📊 Categorical with too many numerical values, converting automatically: {stats}")
                return FeatureType.NUMERIC
        if stats.unique_ratio > MIN_TEXT_UNIQUE_RATIO:
            return FeatureType.TEXT
        elif stats.n_unique >= MIN_TEXT_UNIQUE_FREQUENCY:
            return FeatureType.TEXT
        else:
            return FeatureType.CATEGORICAL
    raise ValueError(f"Unsupported OpenML type: {assumed_type}")
