from typing import Dict, Set

from pandas import DataFrame, Series

from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular.preprocessing.objects import FeatureType
from tabular.preprocessing.textual import replace_unspaced_symbols, normalize_col_name


def curate_features(x: DataFrame, y: Series, curation: CuratedDataset, feature_types: Dict[FeatureType, Set[str]]):
    _validate_raw_names(x=x, y=y, curation=curation)
    _curate_column_names(x=x, curation=curation, feature_types=feature_types)
    _curate_column_values(x=x, curation=curation, feature_types=feature_types)


def assert_no_wrong_key(s: Series, mapper: Dict[str, str]):
    existing_values = {str(k) for k in set(s)}
    wrong_keys = set(mapper).difference(existing_values)
    if wrong_keys:
        raise ValueError(f"❓❓❓ Missing keys for {s.name} in mapping: {wrong_keys}. Existing values: {existing_values}")


def _curate_column_values(x: DataFrame, curation: CuratedDataset, feature_types: Dict[FeatureType, Set[str]]):
    unmappable_types = {FeatureType.NUMERIC, FeatureType.DATE, FeatureType.TEXT}
    unmappable_features = {f for feat_type in unmappable_types for f in feature_types[feat_type]}
    for col in x.columns:
        feat = curation.get_feature(col)
        if feat is None:
            continue
        if feat.new_name in unmappable_features:
            assert not feat.value_mapping, f"Feature {feat.new_name} is not mappable"
            continue
        if not feat.allow_missing_key:
            assert_no_wrong_key(s=x[col], mapper=feat.value_mapping)
        new_values = x[col].apply(lambda v: feat.value_mapping.get(str(v), str(v)))
        new_values = new_values.apply(replace_unspaced_symbols)
        x[col] = new_values


def _validate_raw_names(x: DataFrame, y: Series, curation: CuratedDataset):
    for feat in curation.features:
        if feat.raw_name not in set(x).union({y.name}):
            raise ValueError(f"Feature {feat.raw_name} not found in dataset {curation.name}")


def _curate_column_names(x: DataFrame, curation: CuratedDataset, feature_types: Dict[FeatureType, Set[str]]):
    old2new = curation.name_mapper.copy()
    for col in x.columns:
        if col not in old2new:
            old2new[col] = normalize_col_name(col)
    x.rename(columns=old2new, inplace=True)
    for feat_type in feature_types:
        new_names = {old2new[f] for f in feature_types[feat_type]}
        feature_types[feat_type] = new_names

