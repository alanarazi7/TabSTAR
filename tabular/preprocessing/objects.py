from enum import StrEnum

from tabular.constants import NUM_VERBALIZATION


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


FEAT2EMOJI = {FeatureType.BOOLEAN: "☑️",
              FeatureType.NUMERIC: "🔢",
              FeatureType.CATEGORICAL: "🏷️",
              FeatureType.DATE: "📅",
              FeatureType.TEXT: "📝",
              FeatureType.UNSUPPORTED: "❌"}

class PreprocessingMethod(StrEnum):
    TABSTAR = f"TabSTAR-{NUM_VERBALIZATION}"
    CARTE = "Carte"
    TABPFNV2 = "TabPFN-v2"
    TREES = "Trees"
    TREES_OPT = "Trees-Optuna"
    CATBOOST = "CatBoost"
    CATBOOST_OPT = "CatBoost-Optuna"


CV_METHODS = {PreprocessingMethod.CATBOOST_OPT, PreprocessingMethod.TREES_OPT}