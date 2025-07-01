from dataclasses import dataclass
from typing import Optional, Tuple

from pandas import DataFrame, Series

from tabstar.datasets.all_datasets import TabularDatasetID
from tabstar.preprocessing.dates import series_to_dt
from tabstar.preprocessing.feat_types import convert_series_to_numeric, convert_series_to_textual
from tabstar.preprocessing.texts import normalize_col_name
from tabstar_paper.datasets.curation_mapping import get_curated
from tabstar_paper.datasets.curation_objects import CuratedDataset, CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

@dataclass
class TabularDataset:
    x: DataFrame
    y: Series
    task_type: SupervisedTask
    dataset_id: TabularDatasetID

    @property
    def is_cls(self) -> bool:
        return self.task_type != SupervisedTask.REGRESSION


def curate_dataset(x: DataFrame, y: Optional[Series], dataset_id: TabularDatasetID) -> TabularDataset:
    curation = get_curated(dataset_id)
    x = drop_columns(x, curation=curation)
    x, y = set_target_if_missing(x=x, y=y, curation=curation)
    task_type = curation.target.task_type
    y = curate_target_values(y=y, target=curation.target, task_type=task_type)
    y.name = curation.target.new_name
    x = curate_feature_values(x=x, curation=curation)
    x = curate_column_names(x=x, curation=curation)
    x, y = remove_missing_target_rows(x=x, y=y)
    dataset = TabularDataset(x=x, y=y, task_type=task_type, dataset_id=dataset_id)
    return dataset


def drop_columns(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    if curation.cols_to_drop:
        x = x.drop(columns=curation.cols_to_drop, errors='ignore')
    return x

def set_target_if_missing(x: DataFrame, y: Optional[Series], curation: CuratedDataset) -> Tuple[DataFrame, Series]:
    if y is not None:
        return x, y
    target_var = curation.target.raw_name
    if target_var not in x.columns:
        raise ValueError(f"Target variable {target_var} not found! Have: {x.columns}")
    y = x[target_var].copy()
    x = x.drop(columns=[target_var])
    return x, y

def curate_target_values(y: Series, target: CuratedTarget, task_type: SupervisedTask) -> Series:
    if target.processing_func:
        y = y.apply(target.processing_func)
    if target.numeric_missing:
        assert target.task_type == SupervisedTask.REGRESSION
        y = convert_series_to_numeric(y, missing_value=target.numeric_missing)
    if task_type == SupervisedTask.REGRESSION:
        return y
    mapper = target.label_mapping
    if not mapper:
        return y
    y = y.apply(lambda v: target.label_mapping.get(str(v), str(v)))
    return y

def curate_feature_values(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    # TODO: this feels a bit repetitive, can we import other functions?
    for feat in curation.features:
        col = feat.raw_name
        if col not in x.columns:
            continue
        if feat.processing_func is not None:
            x[col] = x[col].apply(feat.processing_func)
        if feat.value_mapping:
            x[col] = x[col].apply(lambda v: feat.value_mapping.get(str(v), str(v)))
        feat_type = get_curated_feat_type(feat)
        if feat_type is None:
            continue
        if feat_type == FeatureType.NUMERIC:
            missing_value = feat.numeric_missing if feat.numeric_missing else None
            x[col] = convert_series_to_numeric(s=x[col], missing_value=missing_value)
        elif feat_type in {FeatureType.CATEGORICAL, FeatureType.TEXT, FeatureType.BOOLEAN}:
            x[col] = convert_series_to_textual(s=x[col])
        elif feat_type == FeatureType.DATE:
            x[col] = series_to_dt(x[col])
        else:
            raise ValueError(f"Unsupported feature type: {feat_type}")
    return x

def get_curated_feat_type(feat: CuratedFeature) -> Optional[FeatureType]:
    if feat.feat_type is not None:
        return feat.feat_type
    if feat.value_mapping:
        # # The user can be minimalist and provide a mapping, this automatically turns into a non-numeric feature
        return FeatureType.CATEGORICAL
    if feat.numeric_missing is not None:
        return FeatureType.NUMERIC
    return None

def curate_column_names(x: DataFrame, curation: CuratedDataset) -> DataFrame:
    old2new = curation.name_mapper.copy()
    for col in x.columns:
        if col not in old2new:
            old2new[col] = normalize_col_name(col)
    x = x.rename(columns=old2new)
    return x

def remove_missing_target_rows(x: DataFrame, y: Series):
    missing = y.isnull()
    x = x[~missing]
    y = y[~missing]
    return x, y