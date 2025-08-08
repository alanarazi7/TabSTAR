from enum import StrEnum


class FeatureType(StrEnum):
    # Native OpenML Features
    CATEGORICAL = "nominal"
    NUMERIC = "numeric"
    TEXT = "string"
    DATE = "date"
    # Added by us
    BOOLEAN = "binary"
    UNSUPPORTED = "unsupported"


class SupervisedTask(StrEnum):
    # openml.tasks.TaskType doesn't separate binary and multiclass classification, so we redefine it
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"


class PreprocessingMethod(StrEnum):
    TREES_OPT = "Trees-Optuna"
    CATBOOST_OPT = "CatBoost-Optuna"


CV_METHODS = {PreprocessingMethod.CATBOOST_OPT, PreprocessingMethod.TREES_OPT}