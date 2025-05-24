from typing import Optional, Dict, Set, Tuple

from pandas import Series, DataFrame

from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.curation import curate_features
from tabular.preprocessing.feature_type import convert_dtypes
from tabular.preprocessing.nulls import get_valid_values
from tabular.preprocessing.objects import FeatureType, SupervisedTask
from tabular.preprocessing.redundant_variables import drop_redundant_columns
from tabular.preprocessing.sampling import subsample_big_datasets, downsample_multiple_features
from tabular.preprocessing.sparse import densify_objects
from tabular.preprocessing.target import handle_raw_target


def set_target_drop_redundant_downsample_too_big(x: DataFrame, y: Optional[Series], curation: CuratedDataset, sid: str
                                                 ) -> Tuple[DataFrame, Series, SupervisedTask, CuratedDataset]:
    x = drop_redundant_columns(x, curation=curation)
    x, y, task_type = handle_raw_target(x=x, y=y, curation=curation, sid=sid)
    x, y = densify_objects(x, y)
    x, y = subsample_big_datasets(x, y)
    x, curation = downsample_multiple_features(x, curation)
    assert len(x) == len(y) and len(x.columns) == len(set(x.columns)) and y.name not in x.columns
    return x, y, task_type, curation


def create_raw_dataset(x: DataFrame, y: Series, curation: CuratedDataset, desc: str,
                       feat_types: Dict[FeatureType, Set[str]], sid: str, task_type: SupervisedTask,
                       source_name: Optional[str] = None) -> RawDataset:
    convert_dtypes(x, feature_types=feat_types, curation=curation)
    curate_features(x=x, y=y, curation=curation, feature_types=feat_types)
    raw_dataset = RawDataset(sid=sid, x=x, y=y, task_type=task_type, curation=curation, desc=desc,
                             feature_types=feat_types, source_name=source_name)
    raw_dataset.summarize()
    return raw_dataset



def get_series_summary(s: Series, feat_type: Optional[str] = None) -> str:
    if feat_type is None:
        feat_type = s.dtype
    n_unique = set(get_valid_values(s))
    common_values = []
    for v in s.value_counts().index[:10]:
        if isinstance(v, (int, float)):
            v = round(v, 4)
        common_values.append(str(v))
    return f"{s.name} ({feat_type}, {len(n_unique)} distinct): {common_values}"


def get_dataset_description(name: str, url: str, desc: str, x: DataFrame, y: Optional[Series] = None,
                            feat_types: Optional[Dict[str, str]] = None) -> str:
    x.columns = [c.strip() for c in x.columns]
    if feat_types is None:
        feat_types = {col: x[col].dtype for col in x.columns}
        if y is not None:
            feat_types[str(y.name)] = y.dtype
    features = _get_x_features_description_block(x=x, feat_types=feat_types)
    sep = "\n====\n"
    if y is not None:
        feat_type = feat_types[str(y.name)]
        target_var = f"Target Variable: {get_series_summary(s=y, feat_type=feat_type)}"
    else:
        target_var = ""

    strings = [f"Dataset Name: {name}",
               f"Examples: {len(x)}",
               f"URL: {url}",
               f"Description: {desc}",
               target_var,
               f"Features:\n\n{features}"]
    template = sep.join([s for s in strings if s])
    return template


def _get_x_features_description_block(x: DataFrame, feat_types: Dict[str, str]) -> str:
    features = []
    for feat_name, feat_type in feat_types.items():
        if feat_name not in x.columns:
            continue
        feat_rep = get_series_summary(s=x[feat_name], feat_type=feat_type)
        features.append(feat_rep)
    features = "\n".join(features)
    return features


def get_dataframe_types(x: DataFrame) -> Dict[str, FeatureType]:
    feat2types = {FeatureType.NUMERIC: ['int64', 'int', 'float64', 'float'],
                  FeatureType.TEXT: ['object'],
                  FeatureType.BOOLEAN: ['bool'],
                  FeatureType.DATE: ['datetime64[ns]']}
    ret = {}
    for col, dtype in x.dtypes.items():
        feat_type = [k for k, v in feat2types.items() if dtype in v]
        if len(feat_type) != 1:
            raise ValueError(f"Unsupported dtype {dtype} for column {col}")
        ret[col] = feat_type[0]
    return ret