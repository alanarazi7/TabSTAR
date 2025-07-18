from typing import Optional, Dict, Set, Tuple

from pandas import Series, DataFrame

from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.curation import curate_features
from tabular.preprocessing.feature_type import convert_dtypes
from tabular.preprocessing.objects import FeatureType, SupervisedTask
from tabular.preprocessing.redundant_variables import drop_redundant_columns
from tabular.preprocessing.sparse import densify_objects
from tabular.preprocessing.target import handle_raw_target


def set_target_drop_redundant_downsample_too_big(x: DataFrame, y: Optional[Series], curation: CuratedDataset, sid: str
                                                 ) -> Tuple[DataFrame, Series, SupervisedTask, CuratedDataset]:
    x = drop_redundant_columns(x, curation=curation)
    x, y, task_type = handle_raw_target(x=x, y=y, curation=curation, sid=sid)
    x, y = densify_objects(x, y)
    assert len(x) == len(y) and len(x.columns) == len(set(x.columns)) and y.name not in x.columns
    return x, y, task_type, curation


def create_raw_dataset(x: DataFrame, y: Series, curation: CuratedDataset, feat_types: Dict[FeatureType, Set[str]],
                       sid: str, task_type: SupervisedTask) -> RawDataset:
    convert_dtypes(x, feature_types=feat_types, curation=curation)
    curate_features(x=x, y=y, curation=curation, feature_types=feat_types)
    raw_dataset = RawDataset(sid=sid, x=x, y=y, task_type=task_type, curation=curation, feature_types=feat_types)
    raw_dataset.summarize()
    return raw_dataset


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