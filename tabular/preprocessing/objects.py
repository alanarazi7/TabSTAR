from enum import StrEnum

from tabular.constants import NUM_VERBALIZATION


class FeatureType(StrEnum):
    CATEGORICAL = "🏷️ categorical"
    NUMERIC = "🔢 numeric"
    TEXT = "📝 text"
    DATE = "📅 date"
    BOOLEAN = "☑️ boolean"


class SupervisedTask(StrEnum):
    # openml.tasks.TaskType doesn't separate binary and multiclass classification, so we redefine it
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"



CV_METHODS = {PreprocessingMethod.CATBOOST_OPT, PreprocessingMethod.TREES_OPT}