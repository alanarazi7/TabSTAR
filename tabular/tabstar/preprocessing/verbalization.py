from dataclasses import dataclass
from typing import List

from tabular.datasets.properties import DatasetProperties
from tabular.preprocessing.textual import sanitize_text

TARGET_FEATURE = "Target Feature:"
NUMERICAL_TARGET_FEATURE = f"Numerical {TARGET_FEATURE}"
PREDICTIVE_FEATURE = "Predictive Feature:"
FEATURE_VALUE = "Feature Value:"


@dataclass
class Feature:
    name: str
    value: str

    def __post_init__(self):
        assert isinstance(self.value, str), f"Expected string, got {type(self.value)}"
        self.value = sanitize_text(self.value)

    def verbalize(self) -> str:
        feat_name = f"{PREDICTIVE_FEATURE} {self.name}"
        feat_val = f"{FEATURE_VALUE} {self.value}"
        strings = [feat_name, feat_val]
        return _newline_str(strings)


@dataclass
class ClassificationTarget:
    target_name: str
    target_value: str

    def verbalize(self):
        col = f"{TARGET_FEATURE} {self.target_name}"
        val = f"{FEATURE_VALUE} {self.target_value}"
        return _newline_str([col, val])


@dataclass
class RegressionTarget:
    target_name: str

    def verbalize(self):
        return _newline_str([f"{NUMERICAL_TARGET_FEATURE} {self.target_name}"])


def verbalize_feature(column_name: str, column_value: str) -> str:
    feature = Feature(name=column_name, value=column_value)
    return feature.verbalize()

def _verbalize_target_label(target_name: str, target_value: str) -> str:
    return ClassificationTarget(target_name=target_name, target_value=target_value).verbalize()

def _verbalize_target_numerical(target_name: str) -> str:
    return RegressionTarget(target_name=target_name).verbalize()


def _newline_str(blocks: List[str]) -> str:
    return "\n".join([b for b in blocks if b])


def verbalize_target_values(data: DatasetProperties) -> List[str]:
    if data.is_regression:
        assert not data.targets
        return [_verbalize_target_numerical(data.target_name)]
    else:
        return [_verbalize_target_label(data.target_name, v) for v in data.targets]
