from dataclasses import dataclass
from typing import Dict, Set, List, Optional

import pandas as pd

from tabular.benchmarks.all_datasets import BENCHMARKS2DATASETS
from tabular.datasets.manual_curation_obj import CuratedDataset
from tabular.datasets.tabular_datasets import get_sid
from tabular.preprocessing.objects import SupervisedTask, FeatureType
from tabular.utils.utils import cprint

MIN_EXAMPLES = 200
MAX_DATASET_EXAMPLES = 300_000
MAX_FEATURES = 200


@dataclass
class RawDataset:
    sid: str
    x: pd.DataFrame
    y: pd.Series
    task_type: SupervisedTask
    feature_types: Dict[FeatureType, Set[str]]
    curation: CuratedDataset
    desc: str
    source_name: Optional[str] = None

    def __post_init__(self):
        assert len(self.x) == len(self.y)
        self.verify_unique_col_names()
        if len(self.x.columns) > MAX_FEATURES:
            raise ValueError(f"⚠️⚠️⚠️ Dataset {self.sid} has {len(self.x.columns)} features, we allow {MAX_FEATURES=}!")

    def verify_unique_col_names(self):
        all_cols = list(self.x.columns) + [self.y.name]
        if len(all_cols) != len(set(all_cols)):
            raise ValueError(f"Column names must be unique for dataset {self.sid}!")

    def summarize(self):
        feat_cnt = {ft.name[:3]: len(ls) for ft, ls in self.feature_types.items() if len(ls)}
        features = sum(feat_cnt.values())
        summary = f"{self.sid}: {len(self)} samples. {features} Features: {feat_cnt}. Task: {self.task_type}"
        if self.task_type == SupervisedTask.MULTICLASS:
            summary += f" [{len(set(self.y.values))} classes]"
        cprint(summary)

    @property
    def context(self) -> str:
        return self.curation.context

    @property
    def dates(self) -> List[str]:
        return list(self.feature_types[FeatureType.DATE])

    @property
    def textual(self) -> List[str]:
        return list(self.feature_types[FeatureType.TEXT])

    @property
    def numerical(self) -> List[str]:
        return list(self.feature_types[FeatureType.NUMERIC])

    @property
    def bool_cat(self) -> List[str]:
        types = {FeatureType.BOOLEAN, FeatureType.CATEGORICAL}
        return [col for tp in types for col in self.feature_types[tp]]

    @property
    def bool_cat_text(self) -> List[str]:
        types = {FeatureType.BOOLEAN, FeatureType.CATEGORICAL, FeatureType.TEXT}
        return [col for tp in types for col in self.feature_types[tp]]

    def __len__(self) -> int:
        return len(self.y)

    def to_metadata_row(self) -> Dict:
        d_output = 1 if self.task_type == SupervisedTask.REGRESSION else len(self.y.unique())
        d = {
            "name": self.source_name,
            "sid": self.sid,
            "examples": len(self.x),
            "features": len(self.x.columns),
            "task_type": self.task_type.value,
            "d_output": d_output,
            "context": self.context,
        }
        for ft, cols in self.feature_types.items():
            d[f"feature_{ft.name}"] = len(cols)
        for benchmark_name, datasets in BENCHMARKS2DATASETS.items():
            benchmarks_sids = {get_sid(d) for d in datasets}
            is_in = self.sid in benchmarks_sids
            d[f"benchmark_{benchmark_name}"] = is_in
        return d