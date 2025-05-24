from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Self, Dict, Optional

from tabular.datasets.raw_dataset import RawDataset
from tabular.preprocessing.splits import DataSplit
from tabular.preprocessing.objects import PreprocessingMethod, SupervisedTask, FeatureType
from tabular.preprocessing.target import get_label_repr
from tabular.utils.io_handlers import load_json


@dataclass
class DatasetProperties:
    sid: str
    task_type: SupervisedTask
    target_name: str
    processing: PreprocessingMethod
    context: str
    split_sizes: Dict[str, int]
    feat_cnt: Dict[str, int]
    target_summary: str
    cat_col_names: List[str] = field(default_factory=list)
    cat_col_indices: List[int] = field(default_factory=list)
    targets: Optional[List[str]] = None
    idx2text: Dict[int, str] = field(default_factory=dict)
    feat_types: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls,
               raw: RawDataset,
               splits: List[DataSplit],
               processing: PreprocessingMethod,
               feat_cnt: Dict[str, int],
               targets: Optional[List[str]] = None) -> Self:
        split_sizes = dict(Counter([s for s in splits]))
        target_name = str(raw.y.name)
        target_summary = get_label_repr(y=raw.y, task_type=raw.task_type)

        feat_types = {c: str(tp.value) for tp, ls in raw.feature_types.items() for c in ls}
        # TODO: perhaps cat_col_names should become
        cat_col_names = [c for tp, ls in raw.feature_types.items() for c in ls
                         if tp in {FeatureType.BOOLEAN, FeatureType.CATEGORICAL}]
        cat_col_indices = [i for i, c in enumerate(raw.x.columns) if c in cat_col_names]
        return DatasetProperties(
            sid=raw.sid,
            task_type=raw.task_type,
            target_name=target_name,
            feat_cnt=feat_cnt,
            split_sizes=split_sizes,
            processing=processing,
            target_summary=target_summary,
            cat_col_names=cat_col_names,
            cat_col_indices=cat_col_indices,
            targets=targets,
            context=raw.context,
            feat_types=feat_types,
        )

    def __repr__(self):
        feat_cnt = {k.upper()[:3]: v for k, v in self.feat_cnt.items() if v > 0}
        task_summary = f"{self.task_type} [{self.target_summary}]"
        blocks = [self.sid, feat_cnt, self.split_sizes, task_summary]
        return " | ".join([str(b) for b in blocks])

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, path: str) -> Self:
        data = load_json(path)
        obj = DatasetProperties(**data)
        obj.task_type = SupervisedTask(obj.task_type)
        obj.processing = PreprocessingMethod(obj.processing)
        return obj

    @property
    def d_effective_output(self) -> int:
        if self.task_type == SupervisedTask.REGRESSION:
            assert self.targets is None
            return 1
        else:
            return len(self.targets)

    @property
    def is_regression(self) -> bool:
        return self.task_type == SupervisedTask.REGRESSION

    @property
    def is_binary(self) -> bool:
        return self.task_type == SupervisedTask.BINARY

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == SupervisedTask.MULTICLASS

    @property
    def numerical_col_names(self) -> List[str]:
        return [c for c in self.feat_types if self.feat_types[c] == FeatureType.NUMERIC.value]

    @property
    def cat_bool_col_names(self) -> List[str]:
        cat_bool = {FeatureType.BOOLEAN.value, FeatureType.CATEGORICAL.value}
        return [c for c in self.feat_types if self.feat_types[c] in cat_bool]
