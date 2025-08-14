from dataclasses import dataclass, field
from typing import Set

from pandas import DataFrame, Series

from tabstar.preprocessing.nulls import get_valid_values


MIN_TEXT_UNIQUE_RATIO = 0.8
MIN_TEXT_UNIQUE_FREQUENCY = 100


@dataclass
class SemanticFeatureTypes:
    categorical_features: Set[str] = field(default_factory=set)
    text_features: Set[str] = field(default_factory=set)

def classify_semantic_features(x: DataFrame, numerical_features: Set[str]) -> SemanticFeatureTypes:
    string_feat_types = SemanticFeatureTypes()
    for col in x.columns:
        if col in numerical_features:
            continue
        if _is_text_feature(s=x[col]):
            string_feat_types.text_features.add(col)
        else:
            string_feat_types.categorical_features.add(col)
    return string_feat_types

def _is_text_feature(s: Series) -> bool:
    values = get_valid_values(s)
    if not values:
        return False
    n_unique = len(set(values))
    if n_unique >= MIN_TEXT_UNIQUE_FREQUENCY:
        return True
    unique_ratio = n_unique / len(values)
    return unique_ratio >= MIN_TEXT_UNIQUE_RATIO