from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from tabstar.preprocessing.texts import normalize_col_name
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType


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
