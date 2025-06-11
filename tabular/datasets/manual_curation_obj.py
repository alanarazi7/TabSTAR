from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from tabular.preprocessing.objects import FeatureType, SupervisedTask
from tabular.preprocessing.textual import normalize_col_name


@dataclass
class CuratedTarget:
    task_type: SupervisedTask
    raw_name: str
    new_name: Optional[str] = None
    label_mapping: Dict[str, str] = field(default_factory=dict)
    numeric_missing: Optional[str] = None
    processing_func: Optional[Callable] = None

    def __post_init__(self):
        if self.new_name is None:
            self.new_name = self.raw_name
        self.new_name = normalize_col_name(self.new_name)


@dataclass
class CuratedFeature:
    raw_name: str
    new_name: Optional[str] = None
    value_mapping: Dict[str, str] = field(default_factory=dict)
    feat_type: Optional[FeatureType] = None
    allow_missing_key: bool = False
    numeric_missing: Optional[str] = None
    processing_func: Optional[Callable] = None

    def __post_init__(self):
        if self.new_name is None:
            self.new_name = self.raw_name
        self.new_name = normalize_col_name(self.new_name)
        if self.processing_func is not None:
            assert self.feat_type is not None, "Processing function requires feature type to be set"


@dataclass
class CuratedDataset:
    name: str
    target: CuratedTarget
    features: List[CuratedFeature]
    cols_to_drop: List[str]

    @classmethod
    def from_module(cls, module):
        base_name = module.__name__.split('.')[-1]
        return cls(name=base_name,
                   target=module.TARGET,
                   features=module.FEATURES,
                   cols_to_drop=module.COLS_TO_DROP)

    @property
    def name_mapper(self) -> Dict[str, str]:
        return {f.raw_name: f.new_name for f in self.features if f.raw_name != f.new_name}

    def get_feature(self, feat_name: str) -> Optional[CuratedFeature]:
        assert feat_name not in self.cols_to_drop
        feat_list = [f for f in self.features if feat_name in {f.raw_name, f.new_name}]
        if len(feat_list) == 0:
            return None
        assert len(feat_list) == 1
        feat = feat_list[0]
        return feat
